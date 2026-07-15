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
    
    // 1. Look in the same directory as the current executable (debug or release)
    let dev_path = exe_dir.join("whisper_worker");
    if dev_path.exists() {
        return Ok(dev_path);
    }
    
    // 2. Look in the sibling release folder if running in debug (dev mode)
    if let Some(target_dir) = exe_dir.parent() {
        let release_path = target_dir.join("release").join("whisper_worker");
        if release_path.exists() {
            return Ok(release_path);
        }
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

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct ReferenceQA {
    question: String,
    answer: String,
}

struct AppState {
    capture_session: Mutex<Option<audio::CaptureSession>>,
    transcribe_tx: tokio::sync::mpsc::Sender<Vec<f32>>,
    engines: Mutex<Option<models::ModelEngines>>,
    whisper_worker: Mutex<Option<WhisperWorker>>,
    system_prompt: Mutex<String>,
    conversation_history: Mutex<Vec<(String, String)>>,
    active_generation_id: std::sync::atomic::AtomicU64,
    reference_qas: Mutex<Vec<ReferenceQA>>,
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
    *prompt_guard = "You are an expert computer science tutor specializing in Database Management Systems (DBMS). The input query is transcribed from speech and may contain phonetic errors or typos (e.g., 'areas' instead of 'arrays', 'pointer' instead of 'painter'). If a word seems out of context for computer science/programming, contextually correct it to the most relevant computer science term. Then, provide a technically accurate, simple explanation. Keep your answer direct and around 2 to 3 sentences, matching the style and format of these examples:\n\nExample 1:\nUser: What is a Primary Key?\nAssistant: A Primary Key is a unique identifier for a record in a database table. It ensures that no duplicate values exist in the key column and cannot contain NULL values.\n\nExample 2:\nUser: Which SQL statement is used to retrieve data?\nAssistant: The SELECT statement is used to retrieve data from a database. It allows you to specify the columns you want to fetch and filter records using a WHERE clause.".to_string();

    let mut history_guard = state.conversation_history.lock().unwrap();
    history_guard.clear();

    let app_data_dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("Failed to resolve App Data directory: {}", e))?;

    if app_data_dir.exists() {
        fs::remove_dir_all(&app_data_dir)
            .map_err(|e| format!("Failed to delete App Data directory: {}", e))?;
        let _ = fs::create_dir_all(&app_data_dir);
    }

    Ok(())
}

fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let v1: Vec<char> = s1.chars().collect();
    let v2: Vec<char> = s2.chars().collect();
    let len1 = v1.len();
    let len2 = v2.len();
    let mut dp = vec![vec![0; len2 + 1]; len1 + 1];
    
    for i in 0..=len1 {
        dp[i][0] = i;
    }
    for j in 0..=len2 {
        dp[0][j] = j;
    }
    
    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if v1[i - 1] == v2[j - 1] { 0 } else { 1 };
            dp[i][j] = std::cmp::min(
                dp[i - 1][j] + 1,
                std::cmp::min(dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
            );
        }
    }
    dp[len1][len2]
}

fn calculate_similarity(s1: &str, s2: &str) -> f64 {
    let clean1: String = s1.chars().filter(|&c| c.is_alphanumeric() || c.is_whitespace()).collect::<String>().to_lowercase();
    let clean2: String = s2.chars().filter(|&c| c.is_alphanumeric() || c.is_whitespace()).collect::<String>().to_lowercase();
    
    let c1 = clean1.trim();
    let c2 = clean2.trim();
    
    if c1 == c2 {
        return 1.0;
    }
    
    if c1.contains(c2) || c2.contains(c1) {
        let len_diff = (c1.len() as isize - c2.len() as isize).abs();
        if len_diff < 20 {
            return 0.90;
        }
    }
    
    let dist = levenshtein_distance(c1, c2);
    let max_len = std::cmp::max(c1.len(), c2.len());
    if max_len == 0 {
        return 1.0;
    }
    1.0 - (dist as f64 / max_len as f64)
}

#[tauri::command]
async fn load_reference_qas(state: State<'_, AppState>) -> Result<Vec<ReferenceQA>, String> {
    let qas = state.reference_qas.lock().unwrap().clone();
    Ok(qas)
}

#[tauri::command]
async fn save_reference_qas(
    qas: Vec<ReferenceQA>,
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    {
        let mut guard = state.reference_qas.lock().unwrap();
        *guard = qas.clone();
    }
    
    if let Ok(app_data_dir) = app_handle.path().app_data_dir() {
        let _ = fs::create_dir_all(&app_data_dir);
        let ref_file = app_data_dir.join("reference_qas.json");
        let content = serde_json::to_string_pretty(&qas)
            .map_err(|e| format!("Failed to serialize QAs: {}", e))?;
        fs::write(ref_file, content)
            .map_err(|e| format!("Failed to write file: {}", e))?;
    }
    Ok(())
}

#[tauri::command]
fn clear_conversation_history(state: State<'_, AppState>) -> Result<(), String> {
    let mut history_guard = state.conversation_history.lock().unwrap();
    history_guard.clear();
    println!("[AI Engine] Conversation history cleared.");
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

fn sanitize_computer_science_terms(text: &str) -> String {
    // Direct phrase-level normalizations for spaced/hyphenated variations
    let normalized = text
        .replace("N-A", "an array")
        .replace("n-a", "an array")
        .replace("N.A.", "an array")
        .replace("n.a.", "an array");

    let words: Vec<String> = normalized.split_whitespace().map(|w| w.to_string()).collect();
    let mut processed_words = Vec::new();
    let mut i = 0;
    while i < words.len() {
        if i + 1 < words.len() {
            let clean_curr = words[i].chars().filter(|&c| c.is_alphabetic()).collect::<String>().to_lowercase();
            let clean_next = words[i+1].chars().filter(|&c| c.is_alphabetic()).collect::<String>().to_lowercase();
            if clean_curr == "n" && clean_next == "a" {
                processed_words.push("an array".to_string());
                i += 2;
                continue;
            }
        }
        processed_words.push(words[i].clone());
        i += 1;
    }

    for word in &mut processed_words {
        // Extract only letters for matching (keeping punctuation intact)
        let cleaned: String = word.chars().filter(|&c| c.is_alphabetic()).collect::<String>().to_lowercase();
        match cleaned.as_str() {
            "area" => {
                *word = word.replace("area", "array").replace("Area", "Array").replace("AREA", "ARRAY");
            }
            "areas" => {
                *word = word.replace("areas", "arrays").replace("Areas", "Arrays").replace("AREAS", "ARRAYS");
            }
            "painter" => {
                *word = word.replace("painter", "pointer").replace("Painter", "Pointer").replace("PAINTER", "POINTER");
            }
            "painters" => {
                *word = word.replace("painters", "pointers").replace("Painters", "Pointers").replace("PAINTERS", "POINTERS");
            }
            "glass" => {
                *word = word.replace("glass", "class").replace("Glass", "Class").replace("GLASS", "CLASS");
            }
            "glasses" => {
                *word = word.replace("glasses", "classes").replace("Glasses", "Classes").replace("GLASSES", "CLASSES");
            }
            _ => {}
        }
    }
    processed_words.join(" ")
}

fn is_noise_or_filler(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    
    // Check for Whisper tag hallucinations like "*sad music*", "*applause*", etc.
    if (trimmed.starts_with('*') && trimmed.ends_with('*')) 
        || (trimmed.starts_with('[') && trimmed.ends_with(']')) 
        || (trimmed.starts_with('(') && trimmed.ends_with(')')) 
    {
        return true;
    }
    
    let lower = trimmed.to_lowercase();
    
    // Ignore single/double character punctuation artifacts (e.g. "-", ".", "_")
    if lower.len() <= 2 && lower.chars().all(|c| !c.is_alphanumeric()) {
        return true;
    }
    
    // Ignore standalone common filler words/phrases (case insensitive)
    let word_count = lower.split_whitespace().count();
    if word_count <= 2 {
        let filler_words = [
            "okay", "ok", "yes", "yeah", "yep", "no", "nah", 
            "thank you", "thanks", "right", "all right", "sigh", 
            "peace", "hello", "hi", "bye", "ooh", "shh", "sure"
        ];
        // Remove trailing punctuation for matching
        let clean_lower = lower.replace(['.', ',', '?', '!'], "").trim().to_string();
        if filler_words.iter().any(|&f| clean_lower == f) {
            return true;
        }
    }
    
    false
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
                                let raw_trim = text.trim().to_string();
                                const _: () = (); // placeholder/guard helper
                                let mut text_trim = sanitize_computer_science_terms(&raw_trim);

                                // Ignore noise / filler words
                                if is_noise_or_filler(&text_trim) {
                                    println!("[AI Engine] Ignored noise/filler: \"{}\"", text_trim);
                                    continue;
                                }

                                block_counter += 1;
                                let now = chrono::Local::now();
                                let timestamp = now.format("%H:%M:%S").to_string();
                                let word_count = text_trim.split_whitespace().count();
                                
                                println!("[AI Engine] Transcript: \"{}\"", text_trim);

                                // Check if we can find a matching reference Q&A first
                                let ref_qas_guard = state.reference_qas.lock().unwrap();
                                let mut matched_answer: Option<String> = None;
                                let mut matched_question: Option<String> = None;
                                let mut best_sim = 0.0;
                                for ref_qa in &*ref_qas_guard {
                                    let sim = calculate_similarity(&text_trim, &ref_qa.question);
                                    if sim > best_sim {
                                        best_sim = sim;
                                        matched_answer = Some(ref_qa.answer.clone());
                                        matched_question = Some(ref_qa.question.clone());
                                    }
                                }
                                if best_sim < 0.75 {
                                    matched_answer = None;
                                    matched_question = None;
                                } else {
                                    println!("[AI Engine] Match found in Reference QAs (score: {:.2}): \"{}\" -> \"{}\"", best_sim, text_trim, matched_question.as_deref().unwrap_or(""));
                                }
                                drop(ref_qas_guard); // Release lock

                                let mut should_trigger_llm = false;
                                let mut initial_emitted = false;

                                if let (Some(answer), Some(question)) = (matched_answer, matched_question) {
                                    if !answer.trim().is_empty() {
                                        // 1. Emit the matched Q&A immediately (bypassing the LLM entirely)
                                        let payload = serde_json::json!({
                                            "id": block_counter.to_string(),
                                            "timestamp": timestamp,
                                            "text": question,
                                            "answer": Some(answer.clone()),
                                            "isQuestion": true,
                                        });
                                        let _ = app_handle.emit("transcription", payload);
                                        
                                        // 2. Add to conversation history so future turns have context
                                        let mut history_guard = state.conversation_history.lock().unwrap();
                                        history_guard.push((question, answer));
                                        
                                        continue;
                                    } else {
                                        // Dynamic query correction mode: overwrite transcript text with the clean question!
                                        println!("[AI Engine] Correcting transcript using reference question: \"{}\" -> \"{}\"", text_trim, question);
                                        text_trim = question;
                                        
                                        // Emit the corrected question immediately
                                        let payload = serde_json::json!({
                                            "id": block_counter.to_string(),
                                            "timestamp": timestamp,
                                            "text": text_trim.clone(),
                                            "answer": None::<String>,
                                            "isQuestion": false,
                                        });
                                        let _ = app_handle.emit("transcription", payload);
                                        initial_emitted = true;
                                        
                                        should_trigger_llm = true;
                                    }
                                }

                                if !initial_emitted {
                                    // Emit the block to the UI immediately as a non-question (left-column only)
                                    let payload = serde_json::json!({
                                        "id": block_counter.to_string(),
                                        "timestamp": timestamp,
                                        "text": text_trim.clone(),
                                        "answer": None::<String>,
                                        "isQuestion": false,
                                    });
                                    let _ = app_handle.emit("transcription", payload);
                                }

                                if !should_trigger_llm {
                                    // 2. Check if the transcript contains typical question/command markers to trigger LLM
                                    let text_lower = text_trim.to_lowercase();
                                    let is_potential_query = text_lower.contains('?') 
                                        || text_lower.contains("what")
                                        || text_lower.contains("how")
                                        || text_lower.contains("why")
                                        || text_lower.contains("who")
                                        || text_lower.contains("can")
                                        || text_lower.contains("could")
                                        || text_lower.contains("explain")
                                        || text_lower.contains("tell")
                                        || text_lower.contains("describe")
                                        || text_lower.contains("difference")
                                        || text_lower.contains("define")
                                        || text_lower.contains("meaning")
                                        || text_lower.contains("definition")
                                        || text_lower.contains("discuss")
                                        || text_lower.contains("which");

                                    if word_count >= 3 && is_potential_query {
                                        should_trigger_llm = true;
                                    }
                                }

                                if should_trigger_llm {
                                     let system_prompt = state.system_prompt.lock().unwrap().clone();
                                     let app_handle_clone = app_handle.clone();
                                     let text_clone = text_trim.clone();
                                     
                                     // Increment generation ID to cancel any running generation
                                     let my_id = state.active_generation_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                                     
                                     // Run generation on a blocking thread pool
                                     tokio::task::spawn_blocking(move || {
                                         let engines_ref = app_handle_clone.state::<AppState>();
                                         let engines_guard = engines_ref.engines.lock().unwrap();
                                         if let Some(engines) = &*engines_guard {
                                             // Lock history inside the thread pool to get the latest completed conversation history
                                             let history = engines_ref.conversation_history.lock().unwrap().clone();
                                             
                                             let block_id = block_counter.to_string();
                                             let mut answer_accum = String::new();
                                             let res = engines.answer_question_with_history(
                                                 &system_prompt,
                                                 &history,
                                                 &text_clone,
                                                 my_id,
                                                 &engines_ref.active_generation_id,
                                                 |token| {
                                                     answer_accum.push_str(token);
                                                     let token_payload = serde_json::json!({
                                                         "id": block_id.clone(),
                                                         "token": token,
                                                     });
                                                     let _ = app_handle_clone.emit("llm-token", token_payload);
                                                 }
                                             );
                                             
                                             // Only append to history if the generation was not cancelled midway
                                             if engines_ref.active_generation_id.load(std::sync::atomic::Ordering::SeqCst) == my_id {
                                                 let final_answer = answer_accum.trim();
                                                 if !final_answer.is_empty() {
                                                     println!("[AI Engine] Answer: \"{}\"", final_answer);
                                                     // Save the successful exchange to history
                                                     let mut history_guard = engines_ref.conversation_history.lock().unwrap();
                                                     history_guard.push((text_clone, final_answer.to_string()));
                                                 } else {
                                                     println!("[AI Engine] Answer was empty, query ignored.");
                                                 }
                                             } else {
                                                 println!("[AI Engine] Generation ID {} was cancelled, skipping history update.", my_id);
                                             }
                                             
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

            let ref_file = app_data_dir.join("reference_qas.json");
            let loaded_qas = if ref_file.exists() {
                if let Ok(content) = fs::read_to_string(&ref_file) {
                    serde_json::from_str::<Vec<ReferenceQA>>(&content).unwrap_or_default()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            app.manage(AppState {
                 capture_session: Mutex::new(None),
                 transcribe_tx: tx,
                 engines: Mutex::new(loaded_engines),
                 whisper_worker: Mutex::new(loaded_worker),
                 system_prompt: Mutex::new("You are an expert computer science tutor specializing in Database Management Systems (DBMS). The input query is transcribed from speech and may contain phonetic errors or typos (e.g., 'areas' instead of 'arrays', 'pointer' instead of 'painter'). If a word seems out of context for computer science/programming, contextually correct it to the most relevant computer science term. Then, provide a technically accurate, simple explanation. Keep your answer direct and around 2 to 3 sentences. Do not write any greetings or conversational filler.".to_string()),
                 conversation_history: Mutex::new(Vec::new()),
                 active_generation_id: std::sync::atomic::AtomicU64::new(0),
                 reference_qas: Mutex::new(loaded_qas),
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
            request_screen_permission,
            clear_conversation_history,
            load_reference_qas,
            save_reference_qas
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
