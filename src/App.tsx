import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import gsap from "gsap";
import { 
  Play, 
  Square, 
  RotateCcw, 
  Settings, 
  CheckCircle2, 
  Download, 
  Sliders, 
  FileText, 
  X,
  Copy,
  Check,
  Sparkles,
  Cpu
} from "lucide-react";

interface TranscriptionBlock {
  id: string;
  timestamp: string;
  text: string;
  answer?: string;
  isQuestion: boolean;
}

// Module-level trackers to prevent duplicate listeners under HMR or double-mounts
let globalUnsubscribeTranscribe: (() => void) | undefined;
let globalUnsubscribeTokens: (() => void) | undefined;
let globalUnsubscribeProgress: (() => void) | undefined;

function App() {
  // Navigation & UI state
  const [isOnboarded, setIsOnboarded] = useState<boolean>(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [status, setStatus] = useState<"Idle" | "Listening" | "Inferring">("Idle");
  
  // Onboarding specific state
  const [screenPermission, setScreenPermission] = useState<boolean | null>(null);
  const [micPermission, setMicPermission] = useState<boolean | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<{ whisper: number; llm: number }>({
    whisper: 0,
    llm: 0
  });
  const [isDownloading, setIsDownloading] = useState<boolean>(false);

  // Settings state: Optimized for small interview answers
  const [systemPrompt, setSystemPrompt] = useState<string>(
    "You are a helpful assistant. Give a concise, clear answer suitable for a job interview. Keep it to 1-2 short sentences."
  );

  // App running state
  const [blocks, setBlocks] = useState<TranscriptionBlock[]>([]);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // DOM Refs for GSAP list slide down
  const transcriptListRef = useRef<HTMLDivElement>(null);
  const answersListRef = useRef<HTMLDivElement>(null);
  const prevBlocksLength = useRef<number>(0);
  const prevQuestionsLength = useRef<number>(0);

  // GSAP animation on new list items
  useEffect(() => {
    // 1. Slide down left transcript list
    if (blocks.length > prevBlocksLength.current) {
      if (transcriptListRef.current) {
        const firstChild = transcriptListRef.current.firstElementChild;
        if (firstChild) {
          gsap.fromTo(
            firstChild,
            { height: 0, opacity: 0, y: -20, scale: 0.95 },
            { 
              height: "auto", 
              opacity: 1, 
              y: 0, 
              scale: 1,
              duration: 0.6, 
              ease: "power3.out",
              clearProps: "all"
            }
          );
        }
      }
    }
    prevBlocksLength.current = blocks.length;

    // 2. Slide down right Q&A answer cards
    const questionsCount = blocks.filter(b => b.isQuestion).length;
    if (questionsCount > prevQuestionsLength.current) {
      if (answersListRef.current) {
        const firstChild = answersListRef.current.firstElementChild;
        if (firstChild) {
          gsap.fromTo(
            firstChild,
            { height: 0, opacity: 0, y: -20, scale: 0.95 },
            { 
              height: "auto", 
              opacity: 1, 
              y: 0, 
              scale: 1,
              duration: 0.7, 
              ease: "power3.out",
              clearProps: "all"
            }
          );
        }
      }
    }
    prevQuestionsLength.current = questionsCount;
  }, [blocks]);

  // Request permissions placeholder
  const requestPermissions = async () => {
    setScreenPermission(true);
    setMicPermission(true);
  };

  // Model downloader using Tauri backend commands
  const startDownload = async () => {
    setIsDownloading(true);
    try {
      await invoke("download_models");
      await invoke("load_models");
      setIsDownloading(false);
      setIsOnboarded(true);
    } catch (err) {
      console.error("Download failed:", err);
      alert(`Model download failed: ${err}`);
      setIsDownloading(false);
    }
  };

  // Core capture commands
  const handleStartCapture = async () => {
    try {
      await invoke("start_capture");
      setStatus("Listening");
    } catch (err) {
      console.error("Failed to start capture:", err);
      alert(`Error starting capture: ${err}`);
    }
  };

  const handleStopCapture = async () => {
    try {
      await invoke("stop_capture");
      setStatus("Idle");
    } catch (err) {
      console.error("Failed to stop capture:", err);
    }
  };

  const handleRestartCapture = async () => {
    await handleStopCapture();
    setTimeout(() => {
      handleStartCapture();
    }, 300);
  };

  // Save Settings wrapper
  const handleSaveSettings = async () => {
    try {
      await invoke("save_system_prompt", { prompt: systemPrompt });
      setIsSettingsOpen(false);
    } catch (err) {
      console.error("Failed to save prompt:", err);
    }
  };

  // Setup Tauri listeners and startup checks
  useEffect(() => {
    // 1. Initial checks
    const initChecks = async () => {
      try {
        const prompt = await invoke<string>("get_system_prompt");
        setSystemPrompt(prompt);

        const exist = await invoke<boolean>("check_models_exist");
        if (exist) {
          await invoke("load_models");
          setIsOnboarded(true);
        }
      } catch (err) {
        console.error("Startup checks failed:", err);
      }
    };
    initChecks();

    // 2. Listeners
    let active = true;

    // Clean up any previously registered global listeners
    if (globalUnsubscribeTranscribe) {
      globalUnsubscribeTranscribe();
      globalUnsubscribeTranscribe = undefined;
    }
    if (globalUnsubscribeTokens) {
      globalUnsubscribeTokens();
      globalUnsubscribeTokens = undefined;
    }
    if (globalUnsubscribeProgress) {
      globalUnsubscribeProgress();
      globalUnsubscribeProgress = undefined;
    }

    const setupListeners = async () => {
      const unsubTranscribe = await listen<TranscriptionBlock>("transcription", (event) => {
        if (!active) return;
        const newBlock = event.payload;
        setBlocks(prev => {
          if (prev.some(b => b.id === newBlock.id)) {
            return prev;
          }
          return [newBlock, ...prev];
        });
        if (newBlock.isQuestion) {
          setStatus("Inferring");
        }
      });
      if (!active) {
        unsubTranscribe();
      } else {
        globalUnsubscribeTranscribe = unsubTranscribe;
      }

      const unsubTokens = await listen<any>("llm-token", (event) => {
        if (!active) return;
        const { id, token } = event.payload;
        setBlocks(prev => 
          prev.map(block => {
            if (block.id === id) {
              return {
                ...block,
                answer: (block.answer || "") + token,
              };
            }
            return block;
          })
        );
        setStatus("Listening");
      });
      if (!active) {
        unsubTokens();
      } else {
        globalUnsubscribeTokens = unsubTokens;
      }

      const unsubProgress = await listen<any>("download-progress", (event) => {
        if (!active) return;
        const { model, progress } = event.payload;
        setDownloadProgress(prev => ({
          ...prev,
          [model]: progress
        }));
      });
      if (!active) {
        unsubProgress();
      } else {
        globalUnsubscribeProgress = unsubProgress;
      }
    };

    setupListeners();

    return () => {
      active = false;
      if (globalUnsubscribeTranscribe) {
        globalUnsubscribeTranscribe();
        globalUnsubscribeTranscribe = undefined;
      }
      if (globalUnsubscribeTokens) {
        globalUnsubscribeTokens();
        globalUnsubscribeTokens = undefined;
      }
      if (globalUnsubscribeProgress) {
        globalUnsubscribeProgress();
        globalUnsubscribeProgress = undefined;
      }
    };
  }, []);

  const copyToClipboard = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error("Copy failed:", err);
    }
  };

  return (
    <div className="min-h-screen bg-darkblue-950 text-slate-100 flex flex-col font-sans selection:bg-blue-600 selection:text-white relative overflow-hidden">
      
      {/* Background glowing effects */}
      <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-blue-glowing/10 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-cyan-glowing/10 blur-[120px] pointer-events-none" />

      {/* ONBOARDING VIEW */}
      {!isOnboarded ? (
        <div className="flex-1 flex flex-col items-center justify-center p-8 max-w-md mx-auto relative z-10">
          <div className="w-20 h-20 bg-gradient-to-tr from-blue-glowing to-cyan-glowing rounded-2xl flex items-center justify-center shadow-lg shadow-blue-500/20 mb-8 animate-pulse p-[2px]">
            <div className="w-full h-full bg-darkblue-950 rounded-[14px] flex items-center justify-center">
              <Sliders className="w-9 h-9 text-blue-glowing" />
            </div>
          </div>
          
          <h1 className="text-4xl font-extrabold tracking-tight bg-gradient-to-r from-blue-glowing to-cyan-glowing bg-clip-text text-transparent mb-2">
            Sidekick
          </h1>
          <p className="text-slate-400 text-center mb-8 text-sm leading-relaxed">
            Your real-time meeting intelligence assistant. Powered by Whisper STT & Qwen LLM running fully offline.
          </p>

          <div className="w-full bg-darkblue-900/60 border border-darkblue-800/80 backdrop-blur-xl rounded-3xl p-6 mb-6 space-y-6 shadow-2xl">
            <div>
              <h2 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-cyan-glowing" />
                1. System Permissions
              </h2>
              <div className="space-y-3">
                <button 
                  onClick={requestPermissions}
                  className="w-full flex items-center justify-between p-3 bg-darkblue-850/80 hover:bg-darkblue-800 rounded-xl transition-all border border-darkblue-800/50 group"
                >
                  <span className="text-xs font-medium text-slate-300 group-hover:text-white transition-colors">Screen & Audio Recording</span>
                  {screenPermission ? (
                    <span className="text-xs text-emerald-400 font-semibold bg-emerald-500/10 px-2.5 py-1 rounded-lg border border-emerald-500/20">Granted</span>
                  ) : (
                    <span className="text-xs text-blue-glowing font-semibold bg-blue-500/10 px-2.5 py-1 rounded-lg border border-blue-500/20">Grant</span>
                  )}
                </button>
                <button 
                  onClick={requestPermissions}
                  className="w-full flex items-center justify-between p-3 bg-darkblue-850/80 hover:bg-darkblue-800 rounded-xl transition-all border border-darkblue-800/50 group"
                >
                  <span className="text-xs font-medium text-slate-300 group-hover:text-white transition-colors">Microphone Access</span>
                  {micPermission ? (
                    <span className="text-xs text-emerald-400 font-semibold bg-emerald-500/10 px-2.5 py-1 rounded-lg border border-emerald-500/20">Granted</span>
                  ) : (
                    <span className="text-xs text-blue-glowing font-semibold bg-blue-500/10 px-2.5 py-1 rounded-lg border border-blue-500/20">Grant</span>
                  )}
                </button>
              </div>
            </div>

            <div>
              <h2 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                <Download className="w-4 h-4 text-cyan-glowing" />
                2. Local AI Models
              </h2>
              {isDownloading ? (
                <div className="space-y-4 bg-darkblue-950/45 p-4 rounded-2xl border border-darkblue-800/40">
                  <div>
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-slate-400">Whisper Tiny STT</span>
                      <span className="text-cyan-glowing font-mono font-bold">{downloadProgress.whisper}%</span>
                    </div>
                    <div className="w-full bg-darkblue-800 h-2 rounded-full overflow-hidden">
                      <div 
                        className="bg-gradient-to-r from-blue-glowing to-cyan-glowing h-full transition-all duration-300"
                        style={{ width: `${downloadProgress.whisper}%` }}
                      ></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-slate-400">Qwen 1.7B LLM</span>
                      <span className="text-cyan-glowing font-mono font-bold">{downloadProgress.llm}%</span>
                    </div>
                    <div className="w-full bg-darkblue-800 h-2 rounded-full overflow-hidden">
                      <div 
                        className="bg-gradient-to-r from-blue-glowing to-cyan-glowing h-full transition-all duration-300"
                        style={{ width: `${downloadProgress.llm}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              ) : (
                <button
                  disabled={!screenPermission || !micPermission}
                  onClick={startDownload}
                  className="w-full py-3 bg-gradient-to-r from-blue-glowing to-cyan-glowing hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed rounded-xl text-sm font-bold text-darkblue-950 shadow-lg shadow-cyan-500/10 hover:shadow-cyan-500/25 transition-all duration-300"
                >
                  Download Models (~1.2 GB)
                </button>
              )}
            </div>
          </div>
          
          <button 
            onClick={() => setIsOnboarded(true)} 
            className="text-xs text-slate-500 hover:text-slate-350 underline transition-colors"
          >
            Skip onboarding (use existing files)
          </button>
        </div>
      ) : (
        /* MAIN APPLICATION VIEW */
        <div className="flex-1 flex flex-col relative z-10">
          {/* Header */}
          <header className="h-16 border-b border-darkblue-900/60 px-6 flex items-center justify-between bg-darkblue-950/80 backdrop-blur-xl sticky top-0 z-30">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-tr from-blue-glowing to-cyan-glowing rounded-lg flex items-center justify-center p-[1px]">
                <div className="w-full h-full bg-darkblue-950 rounded-[7px] flex items-center justify-center">
                  <Sliders className="w-4 h-4 text-cyan-glowing" />
                </div>
              </div>
              <span className="font-bold text-lg bg-gradient-to-r from-blue-glowing to-cyan-glowing bg-clip-text text-transparent">
                Sidekick
              </span>
              <div className="h-4 w-px bg-darkblue-800 mx-2" />
              <div className="flex items-center gap-2 bg-darkblue-900/60 px-3 py-1 rounded-full border border-darkblue-800/80">
                <span className={`w-2 h-2 rounded-full ${
                  status === "Listening" ? "bg-emerald-400 shadow-[0_0_8px_#34d399] animate-ping" : 
                  status === "Inferring" ? "bg-amber-400 shadow-[0_0_8px_#fbbf24] animate-pulse" : "bg-slate-500"
                }`} />
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">{status}</span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button 
                onClick={handleStartCapture}
                className="p-2.5 bg-blue-600 hover:bg-blue-500 rounded-xl transition-all shadow-lg shadow-blue-600/10 hover:shadow-blue-600/25 flex items-center gap-1.5 text-xs font-semibold text-white border border-blue-500/20"
              >
                <Play className="w-3.5 h-3.5 fill-current" /> Start
              </button>
              <button 
                onClick={handleStopCapture}
                className="p-2.5 bg-darkblue-900/80 hover:bg-darkblue-850 rounded-xl transition-all border border-darkblue-850 flex items-center gap-1.5 text-xs font-semibold text-slate-300 hover:text-white"
              >
                <Square className="w-3.5 h-3.5 text-slate-400 fill-current" /> Stop
              </button>
              <button 
                onClick={handleRestartCapture}
                className="p-2.5 bg-darkblue-900/80 hover:bg-darkblue-850 rounded-xl transition-all border border-darkblue-850 text-slate-400 hover:text-white"
                title="Restart Capture"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
              <button 
                onClick={() => setIsSettingsOpen(true)}
                className="p-2.5 bg-darkblue-900/80 hover:bg-darkblue-850 rounded-xl transition-all border border-darkblue-850 ml-2 text-slate-400 hover:text-white"
              >
                <Settings className="w-3.5 h-3.5" />
              </button>
            </div>
          </header>

          {/* Split Dashboard */}
          <main className="flex-1 grid grid-cols-2 divide-x divide-darkblue-900/60 overflow-hidden h-[calc(100vh-4rem)]">
            
            {/* Left Column: Real-time Transcript */}
            <div className="flex flex-col h-full bg-darkblue-950/40 p-6 space-y-4 overflow-hidden">
              <div className="flex items-center justify-between border-b border-darkblue-900 pb-3 flex-shrink-0">
                <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                  <FileText className="w-4 h-4 text-blue-glowing" /> Real-time Transcript
                </h3>
                <span className="text-[10px] text-slate-500 font-mono bg-darkblue-900 px-2 py-0.5 rounded border border-darkblue-800">16kHz mono</span>
              </div>

              {/* Box View */}
              <div className="flex-1 bg-darkblue-900/20 border border-darkblue-850/80 rounded-2xl p-4 flex flex-col min-h-0 shadow-2xl relative overflow-hidden">
                <div 
                  className="flex-1 overflow-y-auto space-y-4 pr-1 scrollbar-thin scrollbar-thumb-darkblue-800 scrollbar-track-transparent" 
                  ref={transcriptListRef}
                >
                  {blocks.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-4">
                      <div className="w-14 h-14 rounded-2xl bg-darkblue-900 border border-darkblue-800 flex items-center justify-center text-blue-glowing shadow-lg shadow-blue-500/5 animate-pulse">
                        <FileText className="w-6 h-6" />
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-slate-400">Waiting for Speech...</p>
                        <p className="text-xs text-slate-500 mt-1 max-w-[240px] leading-relaxed">
                          Play audio, video or start speaking. Live transcript segments will populate here.
                        </p>
                      </div>
                    </div>
                  ) : (
                    blocks.map((block) => (
                      <div 
                        key={block.id} 
                        className={`p-4 rounded-xl border transition-all duration-300 ${
                          block.isQuestion 
                            ? "bg-blue-950/30 border-blue-500/30 shadow-md shadow-blue-500/5 text-slate-100" 
                            : "bg-darkblue-900/40 border-darkblue-850/60 text-slate-300 hover:border-darkblue-800/80"
                        }`}
                      >
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-[10px] text-slate-500 font-mono font-semibold">{block.timestamp}</span>
                          {block.isQuestion && (
                            <span className="text-[9px] text-cyan-glowing font-bold bg-cyan-500/10 px-2 py-0.5 rounded-md border border-cyan-500/20 uppercase tracking-wider flex items-center gap-1">
                              <Sparkles className="w-2.5 h-2.5" /> Question Detected
                            </span>
                          )}
                        </div>
                        <p className="text-sm leading-relaxed font-medium">{block.text}</p>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>

            {/* Right Column: AI Answers */}
            <div className="flex flex-col h-full bg-darkblue-950/20 p-6 space-y-4 overflow-hidden">
              <div className="flex items-center justify-between border-b border-darkblue-900 pb-3 flex-shrink-0">
                <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-cyan-glowing" /> Instant LLM Answers
                </h3>
                <span className="text-[10px] text-slate-500 font-mono bg-darkblue-900 px-2 py-0.5 rounded border border-darkblue-800">Qwen Offline</span>
              </div>

              <div 
                className="flex-1 overflow-y-auto space-y-4 pr-1 scrollbar-thin scrollbar-thumb-darkblue-800 scrollbar-track-transparent" 
                ref={answersListRef}
              >
                {blocks.filter(b => b.isQuestion).length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-4">
                    <div className="w-14 h-14 rounded-2xl bg-darkblue-900 border border-darkblue-800 flex items-center justify-center text-cyan-glowing shadow-lg shadow-cyan-500/5">
                      <Cpu className="w-6 h-6 animate-pulse" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-slate-400">No Questions Detected Yet</p>
                      <p className="text-xs text-slate-500 mt-1 max-w-[240px] leading-relaxed">
                        When a question is transcribed, the local LLM will automatically generate an answer here.
                      </p>
                    </div>
                  </div>
                ) : (
                  blocks.filter(b => b.isQuestion).map((block) => (
                    <div 
                      key={`ans-${block.id}`} 
                      className="bg-gradient-to-br from-darkblue-900/40 to-blue-950/20 border border-darkblue-850/80 rounded-2xl p-5 space-y-3 relative group hover:border-blue-500/30 transition-all duration-300 shadow-xl"
                    >
                      <div className="flex justify-between items-start gap-4 border-b border-darkblue-850/60 pb-2.5">
                        <div className="text-xs text-blue-glowing font-bold leading-relaxed">
                          Q: "{block.text}"
                        </div>
                        <button 
                          onClick={() => copyToClipboard(block.answer || "", block.id)}
                          className="p-1.5 hover:bg-darkblue-800 rounded-lg text-slate-400 hover:text-white transition-all flex-shrink-0"
                          title="Copy Answer"
                          disabled={!block.answer}
                        >
                          {copiedId === block.id ? (
                            <Check className="w-3.5 h-3.5 text-emerald-400" />
                          ) : (
                            <Copy className="w-3.5 h-3.5" />
                          )}
                        </button>
                      </div>
                      <p className="text-sm leading-relaxed text-slate-200">
                        {block.answer || (
                          <span className="text-slate-500 italic animate-pulse flex items-center gap-1.5">
                            Generating answer...
                          </span>
                        )}
                      </p>
                    </div>
                  ))
                )}
              </div>
            </div>
          </main>
        </div>
      )}

      {/* Settings Modal */}
      {isSettingsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-darkblue-950/80 backdrop-blur-md">
          <div className="w-full max-w-md bg-darkblue-900 border border-darkblue-850 rounded-3xl p-6 shadow-2xl space-y-5 animate-in fade-in zoom-in-95 duration-150 relative z-50">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-slate-200">System Pre-Prompt Settings</h3>
              <button 
                onClick={() => setIsSettingsOpen(false)}
                className="p-1 hover:bg-darkblue-800 rounded-lg text-slate-400 hover:text-white transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-400 block">LLM System Instructions</label>
              <textarea 
                rows={4}
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                className="w-full text-sm bg-darkblue-950 border border-darkblue-850 rounded-xl p-3 text-slate-350 focus:outline-none focus:border-blue-500 focus:text-white transition-colors resize-none"
                placeholder="Enter system prompt guidelines..."
              />
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <button 
                onClick={handleSaveSettings}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-xl text-xs font-semibold text-white transition-all shadow-lg shadow-blue-600/10 hover:shadow-blue-600/25 border border-blue-500/20"
              >
                Save Settings
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

