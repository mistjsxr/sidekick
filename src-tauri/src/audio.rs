use std::sync::{Arc, Mutex};
use screencapturekit::stream::SCStream;
use screencapturekit::stream::SCStreamOutput;
use screencapturekit::stream::SCStreamDelegate;
use screencapturekit::stream::content_filter::SCContentFilter;
use screencapturekit::stream::configuration::SCStreamConfiguration;
use screencapturekit::stream::output_type::SCStreamOutputType;
use screencapturekit::cm::CMSampleBuffer;
use screencapturekit::cm::CMSampleBufferExt;
use screencapturekit::shareable_content::SCShareableContent;
use tokio::sync::mpsc::Sender;

// Silence detection settings
const RMS_THRESHOLD: f32 = 0.001;        // Energy threshold for speech (more sensitive for digital speaker capture)
const SILENCE_DURATION_SEC: f32 = 1.5;   // Silence duration to trigger segment boundary
const MAX_BUFFER_DURATION_SEC: f32 = 15.0; // Max audio chunk duration to prevent overflow

pub struct AudioErrorHandler;

impl SCStreamDelegate for AudioErrorHandler {
    fn did_stop_with_error(&self, error: screencapturekit::error::SCError) {
        eprintln!("[Audio Capture Error] Stream stopped with error: {}", error);
    }
}

pub struct CaptureState {
    pub buffered_audio: Vec<f32>,
    pub is_speaking: bool,
    pub silence_samples: usize,
    pub transcribe_tx: Sender<Vec<f32>>,
}

pub struct AudioStreamHandler {
    pub state: Arc<Mutex<CaptureState>>,
}

impl SCStreamOutput for AudioStreamHandler {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
        if let SCStreamOutputType::Audio = of_type {
            // Retrieve Format Description
            let format_desc = sample.format_description();
            if format_desc.is_none() {
                return;
            }
            let format_desc = format_desc.unwrap();
            
            let channels = format_desc.audio_channel_count().unwrap_or(2) as usize;
            let sample_rate = format_desc.audio_sample_rate().unwrap_or(48000.0) as f32;

            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                println!(
                    "[Audio Format] channels: {}, sample_rate: {}, bits: {:?}, is_float: {}, bytes_per_frame: {:?}, num_buffers: {}",
                    channels,
                    sample_rate,
                    format_desc.audio_bits_per_channel(),
                    format_desc.audio_is_float(),
                    format_desc.audio_bytes_per_frame(),
                    sample.audio_buffer_list().map(|l| l.num_buffers()).unwrap_or(0)
                );
            });

            // Get Audio Buffer List
            let audio_buffer_list = sample.audio_buffer_list();
            if audio_buffer_list.is_none() {
                return;
            }
            let audio_buffer_list = audio_buffer_list.unwrap();
            let num_buffers = audio_buffer_list.num_buffers();
            if num_buffers == 0 {
                return;
            }

            // Extract f32 samples
            let src_samples: Vec<f32> = if num_buffers >= channels && channels > 1 {
                // Non-interleaved: take channel 0 (left channel)
                let buffer = audio_buffer_list.get(0).unwrap();
                let data = buffer.data();
                data.chunks_exact(4)
                    .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                    .collect()
            } else {
                // Interleaved or Mono
                let buffer = audio_buffer_list.get(0).unwrap();
                let raw_bytes = buffer.data();
                let interleaved: Vec<f32> = raw_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                    .collect();
                if channels == 1 {
                    interleaved
                } else {
                    // Extract left channel from interleaved stream
                    interleaved
                        .chunks_exact(channels)
                        .map(|c| c[0])
                        .collect()
                }
            };

            if src_samples.is_empty() {
                return;
            }

            // Diagnostic: Print stats of the first few chunks to see if we're getting valid float PCM data
            static mut DIAGNOSTIC_COUNTER: usize = 0;
            unsafe {
                DIAGNOSTIC_COUNTER += 1;
                if DIAGNOSTIC_COUNTER % 50 == 1 {
                    let min_val = src_samples.iter().fold(f32::INFINITY, |m, &x| m.min(x));
                    let max_val = src_samples.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
                    let avg_val = src_samples.iter().sum::<f32>() / src_samples.len() as f32;
                    let sample_slice = if src_samples.len() > 5 { &src_samples[0..5] } else { &src_samples };
                    println!(
                        "[Audio Diagnostics] Chunk #{}: len={}, min={:.6}, max={:.6}, avg={:.6}, samples={:?}",
                        DIAGNOSTIC_COUNTER,
                        src_samples.len(),
                        min_val,
                        max_val,
                        avg_val,
                        sample_slice
                    );
                }
            }

            // Downsample to 16kHz mono PCM
            let resampled = resample_to_16k(&src_samples, sample_rate);

            // Run Voice Activity Detection (VAD)
            let mut state = self.state.lock().unwrap();
            
            // Calculate RMS energy of this block
            let rms = (resampled.iter().map(|&s| s * s).sum::<f32>() / resampled.len() as f32).sqrt();

            state.buffered_audio.extend_from_slice(&resampled);

            let is_currently_silence = rms < RMS_THRESHOLD;
            if !is_currently_silence {
                state.is_speaking = true;
                state.silence_samples = 0;
            } else {
                state.silence_samples += resampled.len();
            }

            // Truncate silent buffers to avoid holding onto/transcribing pure silence
            if !state.is_speaking {
                let pre_roll_samples = 16000; // 1.0 second at 16kHz
                if state.buffered_audio.len() > pre_roll_samples {
                    let drain_len = state.buffered_audio.len() - pre_roll_samples;
                    state.buffered_audio.drain(0..drain_len);
                }
            }

            // Threshold in samples at 16kHz
            let silence_threshold_samples = (SILENCE_DURATION_SEC * 16000.0) as usize;
            let max_buffer_samples = (MAX_BUFFER_DURATION_SEC * 16000.0) as usize;

            // Trigger transcription if speaking ends (silence detected) or max duration reached
            let trigger_transcription = (state.is_speaking && state.silence_samples >= silence_threshold_samples)
                || (state.buffered_audio.len() >= max_buffer_samples);

            if trigger_transcription {
                let audio_chunk = state.buffered_audio.clone();
                state.buffered_audio.clear();
                state.is_speaking = false;
                state.silence_samples = 0;

                // Send speech chunk asynchronously to the AI engine
                let tx = state.transcribe_tx.clone();
                tauri::async_runtime::spawn(async move {
                    if let Err(e) = tx.send(audio_chunk).await {
                        eprintln!("[VAD] Failed to send audio chunk to transcription channel: {}", e);
                    }
                });
            }
        }
    }
}

// Linear resampling from arbitrary sample rate to 16000 Hz
fn resample_to_16k(src: &[f32], src_rate: f32) -> Vec<f32> {
    let target_rate = 16000.0;
    if (src_rate - target_rate).abs() < 1.0 {
        return src.to_vec();
    }

    let ratio = src_rate / target_rate;
    let target_len = (src.len() as f32 / ratio).floor() as usize;
    let mut dest = Vec::with_capacity(target_len);

    for i in 0..target_len {
        let src_index = i as f32 * ratio;
        let index_floor = src_index.floor() as usize;
        let index_ceil = (index_floor + 1).min(src.len() - 1);
        let weight = src_index - index_floor as f32;

        let sample = (1.0 - weight) * src[index_floor] + weight * src[index_ceil];
        dest.push(sample);
    }
    dest
}

pub struct CaptureSession {
    stream: SCStream,
}

impl CaptureSession {
    pub fn start(transcribe_tx: Sender<Vec<f32>>) -> Result<Self, String> {
        // Fetch shareable display content
        let content = SCShareableContent::get()
            .map_err(|e| format!("Failed to get shareable content: {}", e))?;
        
        let display = content.displays().pop()
            .ok_or_else(|| "No system displays found to capture speaker audio from.".to_string())?;

        let filter = SCContentFilter::create()
            .with_display(&display)
            .with_excluding_windows(&[])
            .build();

        // Configure audio capture
        let config = SCStreamConfiguration::new()
            .with_width(100)
            .with_height(100)
            .with_captures_audio(true)
            .with_excludes_current_process_audio(false);

        let state = Arc::new(Mutex::new(CaptureState {
            buffered_audio: Vec::new(),
            is_speaking: false,
            silence_samples: 0,
            transcribe_tx,
        }));

        let mut stream = SCStream::new_with_delegate(&filter, &config, AudioErrorHandler);
        let handler = AudioStreamHandler { state };
        
        stream.add_output_handler(handler, SCStreamOutputType::Audio);
        stream.start_capture().map_err(|e| format!("Stream start failed: {}", e))?;

        println!("[Audio Capture] System speaker stream started successfully.");

        Ok(CaptureSession { stream })
    }

    pub fn stop(&self) -> Result<(), String> {
        self.stream.stop_capture().map_err(|e| format!("Stream stop failed: {}", e))?;
        println!("[Audio Capture] System speaker stream stopped.");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_audio_capture_properties() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<f32>>(100);
        println!("Starting audio capture test...");
        let session = CaptureSession::start(tx);
        match session {
            Ok(s) => {
                println!("Capture session started successfully!");
                // Wait to see if we get any audio chunk
                let mut received = 0;
                for _ in 0..10 {
                    if let Ok(Some(data)) = tokio::time::timeout(Duration::from_secs(2), rx.recv()).await {
                        println!("Received audio chunk of length: {}", data.len());
                        received += 1;
                        if received >= 3 {
                            break;
                        }
                    } else {
                        println!("Waiting for audio chunk...");
                    }
                }

                s.stop().unwrap();
            }
            Err(e) => {
                println!("Failed to start capture session: {}", e);
            }
        }
    }
}

