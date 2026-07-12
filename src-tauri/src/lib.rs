use std::sync::Mutex;
use tauri::{Emitter, Manager, State};
use std::fs;

mod audio;
mod models;

struct WhisperWorker {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: std::process::ChildStdout,
}

impl WhisperWorker {
    fn spawn(model_path: &std::path::Path) -> Result<Self, String> {
        let worker_bin = get_worker_path()?;
        println!("[Main Process] Spawning Whisper Worker sidecar: {:?}", worker_bin);
        
        let mut child = std::process::Command::new(worker_bin)
            .arg(model_path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn whisper_worker: {}", e))?;
            
        let stdin = child.stdin.take().ok_or("Failed to open stdin for worker")?;
        let stdout = child.stdout.take().ok_or("Failed to open stdout for worker")?;
        
        Ok(WhisperWorker { child, stdin, stdout })
    }
    
    fn transcribe(&mut self, audio_data: &[f32]) -> Result<String, String> {
        use std::io::{Read, Write};
        
        // 1. Send number of samples (u32, 4 bytes)
        let num_samples = audio_data.len() as u32;
        self.stdin.write_all(&num_samples.to_le_bytes())
            .map_err(|e| format!("Failed to write sample count: {}", e))?;
            
        // 2. Send float samples (num_samples * 4 bytes)
        let mut byte_buf = vec![0u8; audio_data.len() * 4];
        for (i, &sample) in audio_data.iter().enumerate() {
            let val_bytes = sample.to_le_bytes();
            let start = i * 4;
            byte_buf[start] = val_bytes[0];
            byte_buf[start + 1] = val_bytes[1];
            byte_buf[start + 2] = val_bytes[2];
            byte_buf[start + 3] = val_bytes[3];
        }
        self.stdin.write_all(&byte_buf)
            .map_err(|e| format!("Failed to write audio samples: {}", e))?;
        self.stdin.flush()
            .map_err(|e| format!("Failed to flush stdin: {}", e))?;
            
        // 3. Read response text length (u32, 4 bytes)
        let mut len_bytes = [0u8; 4];
        self.stdout.read_exact(&mut len_bytes)
            .map_err(|e| format!("Failed to read response length: {}", e))?;
        let text_len = u32::from_le_bytes(len_bytes) as usize;
        
        // 4. Read response string bytes
        let mut text_bytes = vec![0u8; text_len];
        self.stdout.read_exact(&mut text_bytes)
            .map_err(|e| format!("Failed to read response string: {}", e))?;
            
        let text = String::from_utf8(text_bytes)
            .map_err(|e| format!("Invalid UTF-8 from worker: {}", e))?;
            
        Ok(text)
    }
}

impl Drop for WhisperWorker {
    fn drop(&mut self) {
        println!("[Main Process] Stopping Whisper Worker sidecar...");
        let _ = self.child.kill();
    }
}

fn get_worker_path() -> Result<std::path::PathBuf, String> {
    let current_exe = std::env::current_exe().map_err(|e| format!("Failed to get current exe path: {}", e))?;
    let exe_dir = current_exe.parent().ok_or("Failed to get exe directory")?;
    
    // During dev, it will be in the same target/debug folder as sidekick
    let dev_path = exe_dir.join("whisper_worker");
    if dev_path.exists() {
        return Ok(dev_path);
    }
    
    // Fallback search in exe directory in case of name suffixes or bundling
    if let Ok(entries) = std::fs::read_dir(exe_dir) {
        for entry in entries.flatten() {
            let filename = entry.file_name().to_string_lossy().into_owned();
            if filename.starts_with("whisper_worker") {
                return Ok(entry.path());
            }
        }
    }
    
    Err("whisper_worker binary not found. Please build the workspace first.".to_string())
}

struct AppState {
    capture_session: Mutex<Option<audio::CaptureSession>>,
    transcribe_tx: tokio::sync::mpsc::Sender<Vec<f32>>,
    engines: Mutex<Option<models::ModelEngines>>,
    whisper_worker: Mutex<Option<WhisperWorker>>,
    system_prompt: Mutex<String>,
}

#[tauri::command]
async fn start_capture(state: State<'_, AppState>) -> Result<(), String> {
    let mut session_guard = state.capture_session.lock().unwrap();
    if session_guard.is_some() {
        return Err("Capture session is already running.".to_string());
    }

    let session = audio::CaptureSession::start(state.transcribe_tx.clone())?;
    *session_guard = Some(session);
    Ok(())
}

#[tauri::command]
async fn stop_capture(state: State<'_, AppState>) -> Result<(), String> {
    let mut session_guard = state.capture_session.lock().unwrap();
    if let Some(session) = session_guard.take() {
        session.stop()?;
    }
    Ok(())
}

#[tauri::command]
async fn load_models(state: State<'_, AppState>, app_handle: tauri::AppHandle) -> Result<(), String> {
    let mut engines_guard = state.engines.lock().unwrap();
    
    let app_data_dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("Failed to resolve App Data directory: {}", e))?;
    
    let whisper_path = app_data_dir.join("ggml-large-v3-turbo-q8_0.bin");
    let qwen_path = app_data_dir.join("Qwen3.5-2B-Q4_K_M.gguf");

    if !whisper_path.exists() || !qwen_path.exists() {
        return Err("Model files are missing. Please complete onboarding first.".to_string());
    }

    if engines_guard.is_none() {
        let engines = models::ModelEngines::load(&whisper_path, &qwen_path)?;
        *engines_guard = Some(engines);
    }

    let mut worker_guard = state.whisper_worker.lock().unwrap();
    if worker_guard.is_none() {
        let worker = WhisperWorker::spawn(&whisper_path)?;
        *worker_guard = Some(worker);
    }

    Ok(())
}

#[tauri::command]
fn get_system_prompt(state: State<'_, AppState>, app_handle: tauri::AppHandle) -> String {
    // Attempt to load from file first
    if let Ok(app_data_dir) = app_handle.path().app_data_dir() {
        let prompt_file = app_data_dir.join("system_prompt.txt");
        if let Ok(content) = fs::read_to_string(prompt_file) {
            let mut prompt_guard = state.system_prompt.lock().unwrap();
            *prompt_guard = content.clone();
            return content;
        }
    }
    state.system_prompt.lock().unwrap().clone()
}

#[tauri::command]
fn save_system_prompt(state: State<'_, AppState>, app_handle: tauri::AppHandle, prompt: String) -> Result<(), String> {
    let mut prompt_guard = state.system_prompt.lock().unwrap();
    *prompt_guard = prompt.clone();

    if let Ok(app_data_dir) = app_handle.path().app_data_dir() {
        let _ = fs::create_dir_all(&app_data_dir);
        let prompt_file = app_data_dir.join("system_prompt.txt");
        let _ = fs::write(prompt_file, prompt);
    }
    Ok(())
}

#[tauri::command]
async fn delete_models(state: State<'_, AppState>, app_handle: tauri::AppHandle) -> Result<(), String> {
    let mut engines_guard = state.engines.lock().unwrap();
    *engines_guard = None;

    let mut worker_guard = state.whisper_worker.lock().unwrap();
    *worker_guard = None;

    let app_data_dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("Failed to resolve App Data directory: {}", e))?;
    
    let whisper_path = app_data_dir.join("ggml-large-v3-turbo-q8_0.bin");
    let qwen_path = app_data_dir.join("Qwen3.5-2B-Q4_K_M.gguf");

    if whisper_path.exists() {
        fs::remove_file(&whisper_path)
            .map_err(|e| format!("Failed to delete Whisper model: {}", e))?;
    }
    if qwen_path.exists() {
        fs::remove_file(&qwen_path)
            .map_err(|e| format!("Failed to delete Qwen LLM model: {}", e))?;
    }

    Ok(())
}

#[tauri::command]
async fn eject_models(state: State<'_, AppState>) -> Result<(), String> {
    let mut engines_guard = state.engines.lock().unwrap();
    *engines_guard = None;

    let mut worker_guard = state.whisper_worker.lock().unwrap();
    *worker_guard = None;

    Ok(())
}

#[tauri::command]
fn check_models_mounted(state: State<'_, AppState>) -> bool {
    let engines_guard = state.engines.lock().unwrap();
    engines_guard.is_some()
}

#[tauri::command]
async fn reset_app(state: State<'_, AppState>, app_handle: tauri::AppHandle) -> Result<(), String> {
    let mut engines_guard = state.engines.lock().unwrap();
    *engines_guard = None;

    let mut worker_guard = state.whisper_worker.lock().unwrap();
    *worker_guard = None;

    let mut prompt_guard = state.system_prompt.lock().unwrap();
    *prompt_guard = "You are a helpful assistant. Give a concise, clear answer suitable for a job interview. Keep it to 1-2 short sentences.".to_string();

    let app_data_dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("Failed to resolve App Data directory: {}", e))?;

    if app_data_dir.exists() {
        fs::remove_dir_all(&app_data_dir)
            .map_err(|e| format!("Failed to delete App Data directory: {}", e))?;
        let _ = fs::create_dir_all(&app_data_dir);
    }

    Ok(())
}

#[tauri::command]
fn check_screen_permission() -> bool {
    #[cfg(target_os = "macos")]
    {
        if let Ok(content) = screencapturekit::shareable_content::SCShareableContent::get() {
            !content.displays().is_empty()
        } else {
            false
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

#[tauri::command]
fn request_screen_permission() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let _ = Command::new("open")
            .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")
            .spawn();
    }
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<f32>>(100);

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(move |app| {
            let app_handle = app.handle().clone();
            let state_app_handle = app.handle().clone();

            // Run background audio consumer + inference thread
            tauri::async_runtime::spawn(async move {
                let mut block_counter = 0;
                
                while let Some(audio_chunk) = rx.recv().await {
                    let state: State<'_, AppState> = state_app_handle.state();
                    
                    let mut worker_guard = state.whisper_worker.lock().unwrap();
                    if let Some(worker) = &mut *worker_guard {
                        println!("[AI Engine] Transcribing audio chunk ({} samples)...", audio_chunk.len());
                        match worker.transcribe(&audio_chunk) {
                            Ok(text) => {
                                if text.is_empty() {
                                    println!("[AI Engine] Whisper output was empty silence.");
                                    continue;
                                }

                                block_counter += 1;
                                let now = chrono::Local::now();
                                let timestamp = now.format("%H:%M:%S").to_string();

                                // Simple question checking
                                let text_lower = text.to_lowercase();
                                let is_question = text.ends_with('?') || 
                                    text_lower.starts_with("how") ||
                                    text_lower.starts_with("what") ||
                                    text_lower.starts_with("why") ||
                                    text_lower.starts_with("where") ||
                                    text_lower.starts_with("when") ||
                                    text_lower.starts_with("who") ||
                                    text_lower.starts_with("which") ||
                                    text_lower.starts_with("can you") ||
                                    text_lower.starts_with("could you") ||
                                    text_lower.starts_with("is there");

                                println!("[AI Engine] Transcript: \"{}\" (is_question: {})", text, is_question);

                                // 1. Emit the block to the UI
                                let payload = serde_json::json!({
                                    "id": block_counter.to_string(),
                                    "timestamp": timestamp,
                                    "text": text.clone(),
                                    "answer": if is_question { Some("") } else { None },
                                    "isQuestion": is_question,
                                });
                                let _ = app_handle.emit("transcription", payload);

                                // 2. If it is a question, generate LLM answer
                                if is_question {
                                    let system_prompt = state.system_prompt.lock().unwrap().clone();
                                    let app_handle_clone = app_handle.clone();
                                    
                                    // Run LLaMA generation on a blocking thread pool
                                    let text_clone = text.clone();
                                    tokio::task::spawn_blocking(move || {
                                        let engines_ref = app_handle_clone.state::<AppState>();
                                        let engines_guard = engines_ref.engines.lock().unwrap();
                                        if let Some(engines) = &*engines_guard {
                                            let block_id = block_counter.to_string();
                                            let res = engines.answer_question(&system_prompt, &text_clone, |token| {
                                                let token_payload = serde_json::json!({
                                                    "id": block_id.clone(),
                                                    "token": token,
                                                });
                                                let _ = app_handle_clone.emit("llm-token", token_payload);
                                            });
                                            if let Err(e) = res {
                                                eprintln!("[AI Engine] Qwen error: {}", e);
                                            }
                                        }
                                    });
                                }
                            }
                            Err(e) => {
                                eprintln!("[AI Engine] Whisper subprocess error: {}", e);
                            }
                        }
                    } else {
                        println!("[AI Engine] Warning: Audio chunk dropped because models are not loaded.");
                    }
                }
            });

            // Initialize app state
            let app_data_dir = app.handle().path().app_data_dir().unwrap();
            let whisper_path = app_data_dir.join("ggml-large-v3-turbo-q8_0.bin");
            let qwen_path = app_data_dir.join("Qwen3.5-2B-Q4_K_M.gguf");

            // Attempt to load models instantly if they already exist
            let loaded_engines = if whisper_path.exists() && qwen_path.exists() {
                models::ModelEngines::load(&whisper_path, &qwen_path).ok()
            } else {
                None
            };

            let loaded_worker = if loaded_engines.is_some() && whisper_path.exists() {
                WhisperWorker::spawn(&whisper_path).ok()
            } else {
                None
            };

            app.manage(AppState {
                capture_session: Mutex::new(None),
                transcribe_tx: tx,
                engines: Mutex::new(loaded_engines),
                whisper_worker: Mutex::new(loaded_worker),
                system_prompt: Mutex::new("You are a helpful assistant. Give a concise, clear answer suitable for a job interview. Keep it to 1-2 short sentences.".to_string()),
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_capture,
            stop_capture,
            models::check_models_exist,
            models::download_models,
            delete_models,
            eject_models,
            check_models_mounted,
            reset_app,
            load_models,
            get_system_prompt,
            save_system_prompt,
            check_screen_permission,
            request_screen_permission
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                println!("[Main Process] Tauri application exiting, dropping model contexts...");
                let state = app_handle.state::<AppState>();
                
                let mut engines_guard = state.engines.lock();
                if let Ok(ref mut guard) = engines_guard {
                    **guard = None;
                }
                drop(engines_guard);

                let mut worker_guard = state.whisper_worker.lock();
                if let Ok(ref mut guard) = worker_guard {
                    **guard = None;
                }
                drop(worker_guard);
            }
        });
}
