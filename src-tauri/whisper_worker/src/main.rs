use std::env;
use std::io::{self, Read, Write};
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: whisper_worker <model_path>");
        std::process::exit(1);
    }
    let model_path = &args[1];

    eprintln!("[Whisper Worker] Loading model from {}...", model_path);
    let whisper_ctx = WhisperContext::new_with_params(
        model_path,
        WhisperContextParameters::default(),
    )?;
    eprintln!("[Whisper Worker] Model loaded on GPU/Metal successfully.");

    let mut stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // 1. Read number of samples (u32, 4 bytes)
        let mut len_bytes = [0u8; 4];
        if stdin.read_exact(&mut len_bytes).is_err() {
            // EOF or pipe closed, exit cleanly
            break;
        }
        let num_samples = u32::from_le_bytes(len_bytes) as usize;

        // 2. Read raw float samples (num_samples * 4 bytes)
        let mut sample_bytes = vec![0u8; num_samples * 4];
        if let Err(e) = stdin.read_exact(&mut sample_bytes) {
            eprintln!("[Whisper Worker] Error reading audio data bytes: {}", e);
            break;
        }

        // Convert raw bytes to f32 slice
        let mut audio_data = vec![0.0f32; num_samples];
        for i in 0..num_samples {
            let start = i * 4;
            let val_bytes = [
                sample_bytes[start],
                sample_bytes[start + 1],
                sample_bytes[start + 2],
                sample_bytes[start + 3],
            ];
            audio_data[i] = f32::from_le_bytes(val_bytes);
        }

        // 3. Perform transcription
        let mut state = match whisper_ctx.create_state() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[Whisper Worker] Failed to create state: {}", e);
                continue;
            }
        };
        
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_n_threads(4);
        params.set_single_segment(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_initial_prompt("computer science, database management system, DBMS, database, programming, array, structure, function, pointer, class, object, inheritance, encapsulation, polymorphism, compiler, interpreter, CPU, RAM, registers, memory, stack, heap, hardware, operating system, OS");

        let start_time = std::time::Instant::now();
        if let Err(e) = state.full(params, &audio_data) {
            eprintln!("[Whisper Worker] Whisper transcription failed: {}", e);
            continue;
        }
        let whisper_duration = start_time.elapsed();

        let num_segments = state.full_n_segments();
        let mut transcript = String::new();
        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i) {
                if let Ok(segment_text) = segment.to_str() {
                    transcript.push_str(segment_text);
                }
            }
        }
        let transcript = transcript.trim().to_string();
        eprintln!("[Whisper Worker] Transcription took: {:?}, output: \"{}\"", whisper_duration, transcript);

        // 4. Send transcribed text back (length followed by UTF-8 string bytes)
        let text_bytes = transcript.into_bytes();
        let text_len = text_bytes.len() as u32;
        if stdout.write_all(&text_len.to_le_bytes()).is_err() ||
           stdout.write_all(&text_bytes).is_err() ||
           stdout.flush().is_err() {
            eprintln!("[Whisper Worker] Pipe closed while writing output.");
            break;
        }
    }

    Ok(())
}
