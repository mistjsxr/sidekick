use std::fs::File;
use std::io::Write;
use std::path::Path;
use futures_util::StreamExt;
use tauri::{AppHandle, Emitter, Manager};
use whisper_rs::{WhisperContext, WhisperContextParameters};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{LlamaModel, AddBos};
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;

pub struct ModelEngines {
    pub whisper_ctx: WhisperContext,
    pub llama_model: LlamaModel,
    pub llama_backend: LlamaBackend,
}

impl ModelEngines {
    pub fn load(whisper_path: &Path, qwen_path: &Path) -> Result<Self, String> {
        println!("[Inference Engine] Loading Whisper model from {:?}", whisper_path);
        let whisper_ctx = WhisperContext::new_with_params(
            whisper_path.to_str().ok_or("Invalid Whisper path")?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| format!("Failed to load Whisper model: {}", e))?;

        println!("[Inference Engine] Initializing LLaMA Backend...");
        let llama_backend = LlamaBackend::init()
            .map_err(|e| format!("Failed to init LLaMA backend: {}", e))?;

        println!("[Inference Engine] Loading Qwen GGUF model (GPU/Metal offloaded) from {:?}", qwen_path);
        // Offload 99 layers to GPU (ensures all layers of this small model fit on Metal GPU)
        let model_params = LlamaModelParams::default().with_n_gpu_layers(99);
        let llama_model = LlamaModel::load_from_file(
            &llama_backend,
            qwen_path,
            &model_params,
        )
        .map_err(|e| format!("Failed to load LLaMA model: {}", e))?;

        Ok(ModelEngines {
            whisper_ctx,
            llama_model,
            llama_backend,
        })
    }

    pub fn transcribe(&self, audio_data: &[f32]) -> Result<String, String> {
        let mut state = self.whisper_ctx.create_state()
            .map_err(|e| format!("Failed to create Whisper state: {}", e))?;

        let mut params = whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_n_threads(4);
        params.set_single_segment(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state.full(params, audio_data)
            .map_err(|e| format!("Whisper transcription failed: {}", e))?;

        let num_segments = state.full_n_segments();

        let mut transcript = String::new();
        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i) {
                if let Ok(segment_text) = segment.to_str() {
                    transcript.push_str(segment_text);
                }
            }
        }

        Ok(transcript.trim().to_string())
    }

    pub fn answer_question<F>(&self, system_prompt: &str, question: &str, on_token: F) -> Result<(), String>
    where
        F: Fn(&str),
    {
        // Initialize context size
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(1024));

        let mut ctx = self.llama_model.new_context(&self.llama_backend, ctx_params)
            .map_err(|e| format!("Failed to create LLaMA context: {}", e))?;

        // Format prompt using Qwen instruct template
        let prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system_prompt, question
        );

        let tokens = self.llama_model.str_to_token(&prompt, AddBos::Always)
            .map_err(|e| format!("Prompt tokenization failed: {}", e))?;

        let max_tokens = 512;
        let mut batch = LlamaBatch::new(max_tokens, 1);

        // Feed prompt tokens to context batch
        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch.add(token, i as i32, &[0], is_last)
                .map_err(|e| format!("Batch add failed: {}", e))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| format!("Failed to decode prompt: {}", e))?;

        // Loop to generate response tokens
        let mut sampler = LlamaSampler::greedy();
        let mut current_pos = tokens.len() as i32;
        let mut generated_tokens = 0;
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        while generated_tokens < 150 {
            let next_token = sampler.sample(&ctx, (batch.n_tokens() - 1) as i32);
            sampler.accept(next_token);

            if next_token == self.llama_model.token_eos() {
                break;
            }

            if let Ok(piece) = self.llama_model.token_to_piece(next_token, &mut decoder, false, None) {
                on_token(&piece);
            }

            batch.clear();
            batch.add(next_token, current_pos, &[0], true)
                .map_err(|e| format!("Batch add token failed: {}", e))?;

            ctx.decode(&mut batch)
                .map_err(|e| format!("Decode token failed: {}", e))?;

            current_pos += 1;
            generated_tokens += 1;
        }

        Ok(())
    }
}

// Downloader helper utilizing HTTP byte streams
async fn download_file(
    app: &AppHandle,
    url: &str,
    dest_path: &Path,
    model_name: &str,
) -> Result<(), String> {
    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Failed to send request: {}", e))?;

    let total_size = response
        .content_length()
        .ok_or_else(|| format!("Failed to read content length for {}", model_name))?;

    let mut file = File::create(dest_path)
        .map_err(|e| format!("Failed to create file at {:?}: {}", dest_path, e))?;

    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();
    let mut last_emitted = 0.0;

    while let Some(item) = stream.next().await {
        let chunk = item.map_err(|e| format!("Error during download: {}", e))?;
        file.write_all(&chunk)
            .map_err(|e| format!("Write failed: {}", e))?;
        downloaded += chunk.len() as u64;

        let percentage = (downloaded as f64 / total_size as f64) * 100.0;
        if percentage - last_emitted > 1.0 || percentage >= 100.0 {
            last_emitted = percentage;
            let payload = serde_json::json!({
                "model": model_name,
                "progress": percentage.round() as u32,
            });
            let _ = app.emit("download-progress", payload);
        }
    }

    Ok(())
}

#[tauri::command]
pub fn check_models_exist(app_handle: AppHandle) -> bool {
    if let Ok(app_data_dir) = app_handle.path().app_data_dir() {
        let whisper_path = app_data_dir.join("ggml-tiny.en.bin");
        let qwen_path = app_data_dir.join("qwen2.5-1.5b-instruct-q4_k_m.gguf");
        whisper_path.exists() && qwen_path.exists()
    } else {
        false
    }
}

#[tauri::command]
pub async fn download_models(app_handle: AppHandle) -> Result<(), String> {
    let app_data_dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("Failed to resolve App Data directory: {}", e))?;
    
    std::fs::create_dir_all(&app_data_dir)
        .map_err(|e| format!("Failed to create App Data directory: {}", e))?;

    let whisper_path = app_data_dir.join("ggml-tiny.en.bin");
    let qwen_path = app_data_dir.join("qwen2.5-1.5b-instruct-q4_k_m.gguf");

    // 1. Download Whisper
    if !whisper_path.exists() {
        download_file(
            &app_handle,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
            &whisper_path,
            "whisper",
        )
        .await?;
    } else {
        let _ = app_handle.emit("download-progress", serde_json::json!({"model": "whisper", "progress": 100}));
    }

    // 2. Download Qwen GGUF
    if !qwen_path.exists() {
        download_file(
            &app_handle,
            "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            &qwen_path,
            "llm",
        )
        .await?;
    } else {
        let _ = app_handle.emit("download-progress", serde_json::json!({"model": "llm", "progress": 100}));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_load_models() {
        let app_data_dir = PathBuf::from("/Users/mistjs/Library/Application Support/com.sidekick.app");
        let whisper_path = app_data_dir.join("ggml-tiny.en.bin");
        let qwen_path = app_data_dir.join("qwen2.5-1.5b-instruct-q4_k_m.gguf");
        assert!(whisper_path.exists());
        assert!(qwen_path.exists());
        let res = ModelEngines::load(&whisper_path, &qwen_path);
        println!("Load result: {:?}", res.is_ok());
        if let Err(e) = &res {
            println!("Error: {}", e);
        }
        assert!(res.is_ok());
    }
}

