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
  const [downloadProgress, setDownloadProgress] = useState<{ whisper: number; llm: number }>({
    whisper: 0,
    llm: 0
  });
  const [isDownloading, setIsDownloading] = useState<boolean>(false);
  const [modelsExist, setModelsExist] = useState<boolean>(false);
  const [modelsMounted, setModelsMounted] = useState<boolean>(false);

  // Settings state: Optimized for small interview answers
  const [systemPrompt, setSystemPrompt] = useState<string>(
    "You are a helpful assistant. Give a concise, clear answer suitable for a job interview. Keep it to 1-2 short sentences."
  );

  // App running state
  const [blocks, setBlocks] = useState<TranscriptionBlock[]>([]);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [snackbar, setSnackbar] = useState<{ message: string; type: "error" | "info" | "success" } | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<boolean>(false);
  const [showResetConfirm, setShowResetConfirm] = useState<boolean>(false);

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

  // Request permissions by launching OS system preferences
  const requestPermissions = async () => {
    try {
      await invoke("request_screen_permission");
    } catch (err) {
      console.error("Failed to request screen permission:", err);
    }
  };

  // Poll for screen capture permission status when onboarding
  useEffect(() => {
    if (isOnboarded) return;

    const checkPermission = async () => {
      try {
        const hasPermission = await invoke<boolean>("check_screen_permission");
        setScreenPermission(hasPermission);
      } catch (err) {
        console.error("Failed to check screen permission:", err);
      }
    };
    
    checkPermission();
    const interval = setInterval(checkPermission, 1500);
    return () => clearInterval(interval);
  }, [isOnboarded]);

  const showToast = (message: string, type: "error" | "info" | "success" = "info") => {
    setSnackbar({ message, type });
  };

  useEffect(() => {
    if (snackbar) {
      const timer = setTimeout(() => setSnackbar(null), 4000);
      return () => clearTimeout(timer);
    }
  }, [snackbar]);

  // Model downloader using Tauri backend commands
  const startDownload = async () => {
    setIsDownloading(true);
    try {
      await invoke("download_models");
      await invoke("load_models");
      setModelsExist(true);
      setModelsMounted(true);
      setIsDownloading(false);
      setIsOnboarded(true);
      showToast("Models downloaded and loaded successfully!", "success");
    } catch (err) {
      console.error("Download failed:", err);
      showToast(`Model download failed: ${err}`, "error");
      setIsDownloading(false);
    }
  };

  // Core capture commands
  const handleStartCapture = async () => {
    if (!modelsExist) {
      showToast("No local AI models are downloaded. Please open Settings and download the models first.", "error");
      return;
    }
    if (!modelsMounted) {
      showToast("Local AI models are ejected (not loaded in memory). Please open Settings and mount the models first.", "error");
      return;
    }
    try {
      await invoke("start_capture");
      setStatus("Listening");
    } catch (err) {
      console.error("Failed to start capture:", err);
      showToast(`Error starting capture: ${err}`, "error");
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

  const handleRestartCapture = () => {
    setBlocks([]);
    showToast("Conversation cleared", "info");
  };

  const handleDeleteModels = async () => {
    try {
      await invoke("delete_models");
      setModelsExist(false);
      setModelsMounted(false);
      setIsOnboarded(false);
      setIsSettingsOpen(false);
      setShowDeleteConfirm(false);
      showToast("Models deleted successfully", "success");
    } catch (err) {
      console.error("Failed to delete models:", err);
      showToast(`Error deleting models: ${err}`, "error");
    }
  };

  const handleEjectModels = async () => {
    try {
      await invoke("eject_models");
      setModelsMounted(false);
      showToast("Models ejected successfully", "success");
    } catch (err) {
      console.error("Failed to eject models:", err);
      showToast(`Error ejecting models: ${err}`, "error");
    }
  };

  const handleMountModels = async () => {
    try {
      await invoke("load_models");
      setModelsMounted(true);
      showToast("Models mounted successfully", "success");
    } catch (err) {
      console.error("Failed to mount models:", err);
      showToast(`Error mounting models: ${err}`, "error");
    }
  };

  const handleDownloadModelsInSettings = async () => {
    setIsDownloading(true);
    try {
      await invoke("download_models");
      await invoke("load_models");
      setModelsExist(true);
      setModelsMounted(true);
      setIsDownloading(false);
      showToast("Models downloaded and mounted successfully", "success");
    } catch (err) {
      console.error("Download failed:", err);
      showToast(`Model download failed: ${err}`, "error");
      setIsDownloading(false);
    }
  };

  const handleResetApp = async () => {
    try {
      await invoke("reset_app");
      setModelsExist(false);
      setModelsMounted(false);
      setBlocks([]);
      setSystemPrompt("You are a helpful assistant. Give a concise, clear answer suitable for a job interview. Keep it to 1-2 short sentences.");
      setIsOnboarded(false);
      setIsSettingsOpen(false);
      setShowResetConfirm(false);
      showToast("Application data reset successfully", "success");
    } catch (err) {
      console.error("Failed to reset application data:", err);
      showToast(`Reset failed: ${err}`, "error");
    }
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

  useEffect(() => {
    if (isSettingsOpen) {
      setShowDeleteConfirm(false);
      setShowResetConfirm(false);
      const checkStatus = async () => {
        try {
          const exist = await invoke<boolean>("check_models_exist");
          setModelsExist(exist);
          if (exist) {
            const mounted = await invoke<boolean>("check_models_mounted");
            setModelsMounted(mounted);
          } else {
            setModelsMounted(false);
          }
        } catch (err) {
          console.error("Failed to check models status:", err);
        }
      };
      checkStatus();
    }
  }, [isSettingsOpen]);

  // Setup Tauri listeners and startup checks
  useEffect(() => {
    // 1. Initial checks
    const initChecks = async () => {
      try {
        const prompt = await invoke<string>("get_system_prompt");
        setSystemPrompt(prompt);

        const exist = await invoke<boolean>("check_models_exist");
        setModelsExist(exist);
        if (exist) {
          await invoke("load_models");
          setModelsMounted(true);
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
                      <span className="text-slate-400">Whisper Large-v3-Turbo Q8</span>
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
                      <span className="text-slate-400">Qwen3.5 2B LLM</span>
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
                  disabled={!screenPermission}
                  onClick={startDownload}
                  className="w-full py-3 bg-gradient-to-r from-blue-glowing to-cyan-glowing hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed rounded-xl text-sm font-bold text-darkblue-950 shadow-lg shadow-cyan-500/10 hover:shadow-cyan-500/25 transition-all duration-300"
                >
                  Download Models (~2.2 GB)
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
              {status === "Listening" || status === "Inferring" ? (
                <button 
                  onClick={handleStopCapture}
                  className="p-2.5 bg-rose-600 hover:bg-rose-500 rounded-xl transition-all shadow-lg shadow-rose-600/10 hover:shadow-rose-600/25 flex items-center gap-1.5 text-xs font-semibold text-white border border-rose-500/20"
                >
                  <Square className="w-3.5 h-3.5 fill-current" /> Stop
                </button>
              ) : (
                <button 
                  onClick={handleStartCapture}
                  className="p-2.5 bg-blue-600 hover:bg-blue-500 rounded-xl transition-all shadow-lg shadow-blue-600/10 hover:shadow-blue-600/25 flex items-center gap-1.5 text-xs font-semibold text-white border border-blue-500/20"
                >
                  <Play className="w-3.5 h-3.5 fill-current" /> Start
                </button>
              )}
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

      {/* Settings Page View */}
      {isSettingsOpen && (
        <div className="fixed inset-0 z-50 bg-darkblue-950 flex flex-col animate-in fade-in duration-200 overflow-hidden">
          {/* Background glowing effects */}
          <div className="absolute top-[20%] left-[-20%] w-[60%] h-[60%] rounded-full bg-blue-glowing/5 blur-[120px] pointer-events-none" />
          <div className="absolute bottom-[20%] right-[-20%] w-[60%] h-[60%] rounded-full bg-cyan-glowing/5 blur-[120px] pointer-events-none" />

          {/* Header */}
          <header className="h-16 border-b border-darkblue-900/60 px-6 flex items-center justify-between bg-darkblue-950/80 backdrop-blur-xl shrink-0">
            <div className="flex items-center gap-3">
              <span className="font-bold text-lg bg-gradient-to-r from-blue-glowing to-cyan-glowing bg-clip-text text-transparent">
                Settings
              </span>
            </div>
            <button 
              onClick={() => setIsSettingsOpen(false)}
              className="px-4 py-2 bg-darkblue-900 hover:bg-darkblue-850 border border-darkblue-800 rounded-xl text-xs font-semibold text-slate-350 hover:text-white transition-all cursor-pointer"
            >
              Back to App
            </button>
          </header>
          
          {/* Main settings options container */}
          <main className="flex-1 overflow-y-auto w-full max-w-2xl mx-auto p-8 space-y-8 relative scrollbar-thin scrollbar-thumb-darkblue-800 scrollbar-track-transparent">

            <div className="space-y-6 relative z-10">
              <div className="space-y-2">
                <h2 className="text-lg font-bold text-slate-200">System Instructions</h2>
                <p className="text-xs text-slate-400">Configure the prompt parameters that define the local LLM's response behavior.</p>
                <textarea 
                  rows={4}
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  className="w-full text-sm bg-darkblue-900 border border-darkblue-850 rounded-2xl p-4 text-slate-300 focus:outline-none focus:border-blue-500 focus:text-white transition-colors resize-none shadow-inner"
                  placeholder="Enter system prompt guidelines..."
                />
              </div>

              {/* Model Management Section */}
              <div className="space-y-4 pt-6 border-t border-darkblue-900">
                <div>
                  <h2 className="text-lg font-bold text-slate-200">Local AI Models</h2>
                  <p className="text-xs text-slate-400">Download, load, or delete model files running offline on your system.</p>
                </div>

                <div className="bg-darkblue-900/60 p-6 rounded-2xl border border-darkblue-850/60 space-y-4 shadow-xl backdrop-blur-md">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-300 font-medium">Whisper Large-v3-Turbo Q8</span>
                    <div className="flex items-center gap-1.5 font-semibold">
                      {!modelsExist ? (
                        <>
                          <span className="w-1.5 h-1.5 rounded-full bg-slate-500" />
                          <span className="text-slate-400">Not Installed</span>
                        </>
                      ) : modelsMounted ? (
                        <>
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_#34d399]" />
                          <span className="text-emerald-400">Loaded</span>
                        </>
                      ) : (
                        <>
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shadow-[0_0_6px_#fbbf24]" />
                          <span className="text-amber-400">Ejected</span>
                        </>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-300 font-medium">Qwen3.5 2B LLM</span>
                    <div className="flex items-center gap-1.5 font-semibold">
                      {!modelsExist ? (
                        <>
                          <span className="w-1.5 h-1.5 rounded-full bg-slate-500" />
                          <span className="text-slate-400">Not Installed</span>
                        </>
                      ) : modelsMounted ? (
                        <>
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_#34d399]" />
                          <span className="text-emerald-400">Loaded</span>
                        </>
                      ) : (
                        <>
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shadow-[0_0_6px_#fbbf24]" />
                          <span className="text-amber-400">Ejected</span>
                        </>
                      )}
                    </div>
                  </div>

                  {isDownloading && (
                    <div className="space-y-4 pt-4 border-t border-darkblue-850/40">
                      <div>
                        <div className="flex justify-between text-xs mb-1.5">
                          <span className="text-slate-400">Whisper Large-v3-Turbo Q8</span>
                          <span className="text-cyan-glowing font-bold">{downloadProgress.whisper}%</span>
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
                          <span className="text-slate-400">Qwen3.5 2B LLM</span>
                          <span className="text-cyan-glowing font-bold">{downloadProgress.llm}%</span>
                        </div>
                        <div className="w-full bg-darkblue-800 h-2 rounded-full overflow-hidden">
                          <div 
                            className="bg-gradient-to-r from-blue-glowing to-cyan-glowing h-full transition-all duration-300"
                            style={{ width: `${downloadProgress.llm}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {!isDownloading && (
                  <div className="pt-2">
                    {modelsExist ? (
                      modelsMounted ? (
                        <button
                          onClick={handleEjectModels}
                          className="w-full py-3 bg-amber-600/20 hover:bg-amber-600 border border-amber-500/30 hover:border-amber-500 text-amber-350 hover:text-white rounded-xl text-sm font-bold transition-all duration-300 cursor-pointer text-center"
                        >
                          Eject Models (Free Memory)
                        </button>
                      ) : (
                        showDeleteConfirm ? (
                          <div className="bg-rose-950/20 border border-rose-500/25 rounded-2xl p-6 space-y-4">
                            <p className="text-xs text-rose-200 font-semibold leading-relaxed">
                              Are you sure you want to delete the local models? This will free up ~2.2 GB of disk space.
                            </p>
                            <div className="flex gap-3">
                              <button
                                onClick={handleDeleteModels}
                                className="flex-1 py-2.5 bg-rose-600 hover:bg-rose-500 text-white rounded-xl text-sm font-bold transition-all cursor-pointer text-center"
                              >
                                Yes, Delete
                              </button>
                              <button
                                onClick={() => setShowDeleteConfirm(false)}
                                className="flex-1 py-2.5 bg-darkblue-800 hover:bg-darkblue-700 text-slate-350 hover:text-white rounded-xl text-sm font-bold transition-all cursor-pointer text-center"
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        ) : (
                          <div className="flex gap-3">
                            <button
                              onClick={handleMountModels}
                              className="flex-1 py-3 bg-emerald-600/20 hover:bg-emerald-600 border border-emerald-500/30 hover:border-emerald-500 text-emerald-350 hover:text-white rounded-xl text-sm font-bold transition-all duration-300 cursor-pointer text-center"
                            >
                              Mount Models
                            </button>
                            <button
                              onClick={() => setShowDeleteConfirm(true)}
                              className="flex-1 py-3 bg-rose-600/20 hover:bg-rose-600 border border-rose-500/30 hover:border-rose-500 text-rose-350 hover:text-white rounded-xl text-sm font-bold transition-all duration-300 cursor-pointer text-center"
                            >
                              Delete Models
                            </button>
                          </div>
                        )
                      )
                    ) : (
                      <button
                        onClick={handleDownloadModelsInSettings}
                        className="w-full py-3 bg-gradient-to-r from-blue-glowing to-cyan-glowing hover:opacity-90 text-darkblue-950 rounded-xl text-sm font-bold transition-all duration-300 cursor-pointer text-center"
                      >
                        Download Models (~2.2 GB)
                      </button>
                    )}
                  </div>
                )}
              </div>

              <div className="flex justify-end gap-3 pt-6 border-t border-darkblue-900">
                <button 
                  onClick={handleSaveSettings}
                  className="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded-xl text-sm font-semibold text-white transition-all shadow-lg shadow-blue-600/10 hover:shadow-blue-600/25 border border-blue-500/20 cursor-pointer"
                >
                  Save Settings
                </button>
              </div>

              {/* Danger Zone Section */}
              <div className="space-y-4 pt-6 border-t border-darkblue-900">
                <div>
                  <h2 className="text-lg font-bold text-rose-400">Danger Zone</h2>
                  <p className="text-xs text-slate-400">Reset the application back to its original factory state.</p>
                </div>

                <div className="bg-rose-950/5 border border-rose-500/10 p-6 rounded-2xl space-y-4">
                  {showResetConfirm ? (
                    <div className="space-y-4">
                      <div className="bg-rose-950/20 border border-rose-500/25 rounded-2xl p-5 space-y-3">
                        <h4 className="text-sm font-bold text-rose-300">Confirm Application Reset</h4>
                        <p className="text-xs text-rose-200 leading-relaxed font-semibold">
                          Are you sure you want to restore the app? This will completely wipe all downloaded model files (~2.2 GB), reset your custom prompts, and return you back to onboarding. This action is irreversible.
                        </p>
                      </div>
                      <div className="flex gap-3">
                        <button
                          onClick={handleResetApp}
                          className="flex-1 py-3 bg-rose-600 hover:bg-rose-500 text-white rounded-xl text-sm font-bold transition-all cursor-pointer text-center"
                        >
                          Yes, Reset Everything
                        </button>
                        <button
                          onClick={() => setShowResetConfirm(false)}
                          className="flex-1 py-3 bg-darkblue-900 hover:bg-darkblue-800 border border-darkblue-800 text-slate-350 hover:text-white rounded-xl text-sm font-bold transition-all cursor-pointer text-center"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    <button
                      onClick={() => setShowResetConfirm(true)}
                      className="w-full py-3 bg-rose-600/10 hover:bg-rose-600 border border-rose-500/20 hover:border-rose-500 text-rose-400 hover:text-white rounded-xl text-sm font-bold transition-all duration-300 cursor-pointer text-center"
                    >
                      Restore App (Factory Reset)
                    </button>
                  )}
                </div>
              </div>
            </div>
          </main>
        </div>
      )}

      {/* Toast Notification */}
      {snackbar && (
        <div className="fixed bottom-6 right-6 z-50 animate-in slide-in-from-bottom-5 fade-in duration-300">
          <div className={`px-4 py-3 rounded-xl shadow-2xl border text-sm font-semibold flex items-center gap-2 backdrop-blur-md ${
            snackbar.type === "error" ? "bg-rose-950/80 border-rose-500/30 text-rose-200" :
            snackbar.type === "success" ? "bg-emerald-950/80 border-emerald-500/30 text-emerald-200" :
            "bg-blue-950/80 border-blue-500/30 text-blue-200"
          }`}>
            <span className={`w-2 h-2 rounded-full ${
              snackbar.type === "error" ? "bg-rose-400 animate-pulse" :
              snackbar.type === "success" ? "bg-emerald-400 animate-pulse" : "bg-blue-400 animate-pulse"
            }`} />
            {snackbar.message}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

