use std::sync::{Arc, Mutex};
use screencapturekit::cm_sample_buffer::CMSampleBuffer;
use screencapturekit::sc_content_filter::{SCContentFilter, InitParams};
use screencapturekit::sc_error_handler::StreamErrorHandler;
use screencapturekit::sc_output_handler::{SCStreamOutputType, StreamOutput};
use screencapturekit::sc_shareable_content::SCShareableContent;
use screencapturekit::sc_stream::SCStream;
use screencapturekit::sc_stream_configuration::SCStreamConfiguration;
use tokio::sync::mpsc::Sender;

// Silence detection settings
const RMS_THRESHOLD: f32 = 0.003;        // Energy threshold for speech
const SILENCE_DURATION_SEC: f32 = 1.5;   // Silence duration to trigger segment boundary
const MAX_BUFFER_DURATION_SEC: f32 = 15.0; // Max audio chunk duration to prevent overflow

pub struct AudioErrorHandler;

impl StreamErrorHandler for AudioErrorHandler {
    fn on_error(&self) {
        eprintln!("[Audio Capture Error] Stream encountered an error.");
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

impl StreamOutput for AudioStreamHandler {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
        if let SCStreamOutputType::Audio = of_type {
            // Retrieve Format Description
            let format_desc = sample.sys_ref.get_format_description();
            if format_desc.is_none() {
                return;
            }
            
            let asbd = format_desc.unwrap().audio_format_description_get_stream_basic_description().copied();
            if asbd.is_none() {
                return;
            }
            let asbd = asbd.unwrap();

            let channels = asbd.channels_per_frame as usize;
            let sample_rate = asbd.sample_rate as f32;

            // Get Raw Audio Buffer List
            let buffers = sample.sys_ref.get_av_audio_buffer_list();
            if buffers.is_empty() {
                return;
            }

            // Extract f32 samples
            let src_samples: Vec<f32> = if buffers.len() >= channels && channels > 1 {
                // Non-interleaved: take channel 0 (left channel)
                let data = &buffers[0].data;
                data.chunks_exact(4)
                    .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                    .collect()
            } else {
                // Interleaved or Mono
                let raw_bytes = &buffers[0].data;
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
                tokio::spawn(async move {
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
        let mut content = SCShareableContent::try_current()
            .map_err(|e| format!("Failed to get shareable content: {}", e))?;
        
        let display = content.displays.pop()
            .ok_or_else(|| "No system displays found to capture speaker audio from.".to_string())?;

        let filter = SCContentFilter::new(InitParams::Display(display));

        // Configure audio capture
        let config = SCStreamConfiguration {
            width: 100,
            height: 100,
            captures_audio: true,
            excludes_current_process_audio: false,
            ..Default::default()
        };

        let state = Arc::new(Mutex::new(CaptureState {
            buffered_audio: Vec::new(),
            is_speaking: false,
            silence_samples: 0,
            transcribe_tx,
        }));

        let mut stream = SCStream::new(filter, config, AudioErrorHandler);
        let handler = AudioStreamHandler { state };
        
        stream.add_output(handler, SCStreamOutputType::Audio);
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
