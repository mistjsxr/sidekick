use std::sync::Mutex;
use tauri::{Emitter, Manager, State};

mod audio;

struct AppState {
    capture_session: Mutex<Option<audio::CaptureSession>>,
    transcribe_tx: tokio::sync::mpsc::Sender<Vec<f32>>,
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<f32>>(100);

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(move |app| {
            let app_handle = app.handle().clone();

            // Run background audio processing consumer
            tauri::async_runtime::spawn(async move {
                let mut chunk_counter = 0;
                while let Some(audio_chunk) = rx.recv().await {
                    chunk_counter += 1;
                    println!("[AI Engine] Received audio chunk {} (size: {} samples)", chunk_counter, audio_chunk.len());

                    // Format current local time for block timestamps
                    let now = chrono::Local::now();
                    let timestamp = now.format("%H:%M:%S").to_string();

                    // For Phase 2 verification, simulate a text question / answer block
                    let is_question = chunk_counter % 2 != 0;
                    let text = if is_question {
                        format!("Question {}: How does Apple Silicon GPU process Whisper models under Metal?", chunk_counter)
                    } else {
                        format!("Statement {}: Meeting audio captured successfully using macOS ScreenCaptureKit API.", chunk_counter)
                    };

                    let answer = if is_question {
                        Some(format!("Glance Answer: Metal allows Whisper to run entirely on the Unified Memory GPU, bypassing CPU-GPU transfer bottlenecks for real-time downsampled 16kHz audio."))
                    } else {
                        None
                    };

                    let payload = serde_json::json!({
                        "id": chunk_counter.to_string(),
                        "timestamp": timestamp,
                        "text": text,
                        "answer": answer,
                        "isQuestion": is_question,
                    });

                    if let Err(e) = app_handle.emit("transcription", payload) {
                        eprintln!("[AI Engine] Failed to emit event: {}", e);
                    }
                }
            });

            app.manage(AppState {
                capture_session: Mutex::new(None),
                transcribe_tx: tx,
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![start_capture, stop_capture])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
