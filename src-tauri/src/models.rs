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
use llama_cpp_2::token::logit_bias::LlamaLogitBias;
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
                    let len = self.buffer.len();
                    if len > 8 {
                        let mut drain_pos = len - 7;
                        while !self.buffer.is_char_boundary(drain_pos) {
                            drain_pos += 1;
                        }
                        if drain_pos < len {
                            self.buffer = self.buffer[drain_pos..].to_string();
                        }
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
                    let mut match_len = 0;
                    for i in (1..=6).rev() {
                        if let Some(suffix) = self.buffer.get(self.buffer.len().saturating_sub(i)..) {
                            if "<think>".starts_with(suffix) {
                                possible_start = true;
                                match_len = suffix.len();
                                break;
                            }
                        }
                    }

                    if possible_start {
                        let emit_len = self.buffer.len() - match_len;
                        if emit_len > 0 {
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
    pub think_token_ids: Vec<llama_cpp_2::token::LlamaToken>,
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

        // Scan the model vocabulary for exact special thinking tokens
        let mut think_token_ids = Vec::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let n_vocab = llama_model.n_vocab();
        for i in 0..n_vocab {
            let token = llama_cpp_2::token::LlamaToken(i as i32);
            if let Ok(piece) = llama_model.token_to_piece(token, &mut decoder, true, None) {
                if piece == "<think>" || piece == "</think>" {
                    println!("[Inference Engine] Found special thinking token: {:?} (ID: {})", piece, i);
                    think_token_ids.push(token);
                }
            }
        }

        Ok(ModelEngines {
            llama_model,
            llama_backend,
            think_token_ids,
        })
    }

    pub fn answer_question_with_history<F>(
        &self,
        system_prompt: &str,
        history: &[(String, String)],
        question: &str,
        my_id: u64,
        active_id: &std::sync::atomic::AtomicU64,
        mut on_token: F,
    ) -> Result<(), String>
    where
        F: FnMut(&str),
    {
        // Initialize context size
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(2048));

        let mut ctx = self.llama_model.new_context(&self.llama_backend, ctx_params)
            .map_err(|e| format!("Failed to create LLaMA context: {}", e))?;

        // Format prompt using Qwen instruct template with history
        let mut prompt = String::new();
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(system_prompt);
        prompt.push_str("\n\nIMPORTANT: Write a direct, technically accurate, and clear answer of 2-3 sentences explaining the query. Start your answer directly with the explanation. Do not write any thinking process, internal monologue, greetings, or conversational filler.");
        prompt.push_str("<|im_end|>\n");
        
        // Limit history to the last 5 entries to prevent context window overflow
        let history_limit = 5;
        let history_start = if history.len() > history_limit {
            history.len() - history_limit
        } else {
            0
        };
        for (q, a) in &history[history_start..] {
            prompt.push_str("<|im_start|>user\n");
            prompt.push_str(q);
            prompt.push_str("<|im_end|>\n");
            prompt.push_str("<|im_start|>assistant\n");
            prompt.push_str(a);
            prompt.push_str("<|im_end|>\n");
        }
        
        prompt.push_str("<|im_start|>user\n");
        prompt.push_str(question);
        prompt.push_str("<|im_end|>\n");
        prompt.push_str("<|im_start|>assistant\n<think>\n</think>\n");

        let tokens = self.llama_model.str_to_token(&prompt, AddBos::Never)
            .map_err(|e| format!("Prompt tokenization failed: {}", e))?;

        let max_tokens = tokens.len();
        let mut batch = LlamaBatch::new(max_tokens, 1);

        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch.add(token, i as i32, &[0], is_last)
                .map_err(|e| format!("Batch add failed: {}", e))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| format!("Failed to decode prompt: {}", e))?;

        // Set up sampler (with temperature to prevent loops on repeating sentences)
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.1),
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::dist(0),
        ]);
        let mut current_pos = tokens.len() as i32;
        let mut generated_tokens = 0;
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut filter = ThinkingFilter::new();

        while generated_tokens < 1024 {
            if active_id.load(std::sync::atomic::Ordering::SeqCst) != my_id {
                println!("\n[AI Engine] Generation cancelled: active_id changed to {}", active_id.load(std::sync::atomic::Ordering::SeqCst));
                break;
            }

            let next_token = sampler.sample(&ctx, (batch.n_tokens() - 1) as i32);
            sampler.accept(next_token);

            if next_token == self.llama_model.token_eos() {
                break;
            }

            if let Ok(piece) = self.llama_model.token_to_piece(next_token, &mut decoder, false, None) {
                // Log raw output to stdout for diagnostics
                print!("{}", piece);
                let _ = std::io::stdout().flush();
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
        println!();
        let _ = std::io::stdout().flush();

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

