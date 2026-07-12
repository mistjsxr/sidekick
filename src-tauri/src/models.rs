use std::fs::File;
use std::io::Write;
use std::path::Path;
use futures_util::StreamExt;
use tauri::{AppHandle, Emitter, Manager};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{LlamaModel, AddBos};
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;

struct ThinkingFilter {
    buffer: String,
    in_think: bool,
}

impl ThinkingFilter {
    fn new() -> Self {
        ThinkingFilter {
            buffer: String::new(),
            in_think: false,
        }
    }

    fn process<F>(&mut self, token: &str, on_token: &mut F)
    where
        F: FnMut(&str),
    {
        self.buffer.push_str(token);

        loop {
            if self.in_think {
                if let Some(pos) = self.buffer.find("</think>") {
                    self.buffer = self.buffer[pos + 8..].to_string();
                    self.in_think = false;
                } else {
                    if self.buffer.len() > 8 {
                        let drain_len = self.buffer.len() - 7;
                        self.buffer = self.buffer[drain_len..].to_string();
                    }
                    break;
                }
            } else {
                if let Some(pos) = self.buffer.find("<think>") {
                    if pos > 0 {
                        on_token(&self.buffer[..pos]);
                    }
                    self.buffer = self.buffer[pos + 7..].to_string();
                    self.in_think = true;
                } else {
                    let mut possible_start = false;
                    for i in 1..=6 {
                        if self.buffer.len() >= i {
                            let suffix = &self.buffer[self.buffer.len() - i..];
                            if "<think>".starts_with(suffix) {
                                possible_start = true;
                                break;
                            }
                        }
                    }

                    if possible_start {
                        if self.buffer.len() > 6 {
                            let emit_len = self.buffer.len() - 6;
                            on_token(&self.buffer[..emit_len]);
                            self.buffer = self.buffer[emit_len..].to_string();
                        }
                        break;
                    } else {
                        on_token(&self.buffer);
                        self.buffer.clear();
                        break;
                    }
                }
            }
        }
    }

    fn flush<F>(self, on_token: &mut F)
    where
        F: FnMut(&str),
    {
        if !self.in_think && !self.buffer.is_empty() {
            on_token(&self.buffer);
        }
    }
}

pub struct ModelEngines {
    pub llama_model: LlamaModel,
    pub llama_backend: LlamaBackend,
}

impl ModelEngines {
    pub fn load(whisper_path: &Path, qwen_path: &Path) -> Result<Self, String> {
        // whisper_path is unused here but kept for signature compatibility during transition
        let _ = whisper_path;

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
            llama_model,
            llama_backend,
        })
    }

    pub fn answer_question<F>(&self, system_prompt: &str, question: &str, mut on_token: F) -> Result<(), String>
    where
        F: FnMut(&str),
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
        let mut filter = ThinkingFilter::new();

        while generated_tokens < 150 {
            let next_token = sampler.sample(&ctx, (batch.n_tokens() - 1) as i32);
            sampler.accept(next_token);

            if next_token == self.llama_model.token_eos() {
                break;
            }

            if let Ok(piece) = self.llama_model.token_to_piece(next_token, &mut decoder, false, None) {
                filter.process(&piece, &mut on_token);
            }

            batch.clear();
            batch.add(next_token, current_pos, &[0], true)
                .map_err(|e| format!("Batch add token failed: {}", e))?;

            ctx.decode(&mut batch)
                .map_err(|e| format!("Decode token failed: {}", e))?;

            current_pos += 1;
            generated_tokens += 1;
        }

        filter.flush(&mut on_token);

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
        let whisper_path = app_data_dir.join("ggml-large-v3-turbo-q8_0.bin");
        let qwen_path = app_data_dir.join("Qwen3.5-2B-Q4_K_M.gguf");
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

    let whisper_path = app_data_dir.join("ggml-large-v3-turbo-q8_0.bin");
    let qwen_path = app_data_dir.join("Qwen3.5-2B-Q4_K_M.gguf");

    // 1. Download Whisper
    if !whisper_path.exists() {
        download_file(
            &app_handle,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin",
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
            "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
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
        let whisper_path = app_data_dir.join("ggml-large-v3-turbo-q8_0.bin");
        let qwen_path = app_data_dir.join("Qwen3.5-2B-Q4_K_M.gguf");
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

