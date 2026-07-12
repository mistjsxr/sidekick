import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { 
  Play, 
  Square, 
  RotateCcw, 
  Settings, 
  CheckCircle2, 
  Download, 
  Sliders, 
  FileText, 
  HelpCircle,
  X
} from "lucide-react";

interface TranscriptionBlock {
  id: string;
  timestamp: string;
  text: string;
  answer?: string;
  isQuestion: boolean;
}

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

  // Settings state
  const [systemPrompt, setSystemPrompt] = useState<string>(
    "You are an assistant. Answer the following question in one sentence:"
  );

  // App running state
  const [blocks, setBlocks] = useState<TranscriptionBlock[]>([
    {
      id: "1",
      timestamp: "10:42:15",
      text: "How do we implement GPU acceleration in Apple Silicon?",
      answer: "Apple Silicon uses the Metal framework for GPU acceleration, allowing unified memory access between the CPU and GPU for extremely fast tensor operations.",
      isQuestion: true
    },
    {
      id: "2",
      timestamp: "10:41:02",
      text: "Welcome to the Glance meeting. We will start by discussing the project scaffolding.",
      isQuestion: false
    }
  ]);

  // Request permissions placeholder
  const requestPermissions = async () => {
    setScreenPermission(true);
    setMicPermission(true);
  };

  // Model download simulator
  const startDownload = () => {
    setIsDownloading(true);
    let whisperVal = 0;
    let llmVal = 0;

    const interval = setInterval(() => {
      if (whisperVal < 100) {
        whisperVal += 5;
        setDownloadProgress(prev => ({ ...prev, whisper: Math.min(whisperVal, 100) }));
      } else if (llmVal < 100) {
        llmVal += 2.5;
        setDownloadProgress(prev => ({ ...prev, llm: Math.min(llmVal, 100) }));
      } else {
        clearInterval(interval);
        setIsDownloading(false);
        setIsOnboarded(true);
      }
    }, 100);
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

  // Setup Tauri listener on mount
  useEffect(() => {
    let unlisten: (() => void) | undefined;

    listen<TranscriptionBlock>("transcription", (event) => {
      const newBlock = event.payload;
      setBlocks(prev => {
        if (prev.some(b => b.id === newBlock.id)) {
          return prev;
        }
        return [newBlock, ...prev];
      });

      if (newBlock.isQuestion) {
        setStatus("Inferring");
        setTimeout(() => setStatus("Listening"), 2500);
      }
    }).then(unlistenFn => {
      unlisten = unlistenFn;
    });

    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      
      {/* ONBOARDING VIEW */}
      {!isOnboarded ? (
        <div className="flex-1 flex flex-col items-center justify-center p-8 max-w-md mx-auto">
          <div className="w-16 h-16 bg-gradient-to-tr from-indigo-500 to-purple-500 rounded-2xl flex items-center justify-center shadow-lg shadow-indigo-500/20 mb-8 animate-pulse">
            <Sliders className="w-8 h-8 text-white" />
          </div>
          
          <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">
            Glance
          </h1>
          <p className="text-slate-400 text-center mb-8 text-sm leading-relaxed">
            Real-time meeting assistant. Captures system speakers, transcribes with Whisper, and provides answers locally via LLM.
          </p>

          <div className="w-full bg-slate-900 border border-slate-800 rounded-2xl p-6 mb-6 space-y-6">
            <div>
              <h2 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-indigo-400" />
                1. System Permissions
              </h2>
              <div className="space-y-3">
                <button 
                  onClick={requestPermissions}
                  className="w-full flex items-center justify-between p-3 bg-slate-800 hover:bg-slate-700/80 rounded-xl transition-all border border-slate-700/50"
                >
                  <span className="text-xs font-medium text-slate-300">Screen & Audio Recording</span>
                  {screenPermission ? (
                    <span className="text-xs text-emerald-400 font-semibold bg-emerald-500/10 px-2.5 py-1 rounded-full">Granted</span>
                  ) : (
                    <span className="text-xs text-indigo-400 font-semibold bg-indigo-500/10 px-2.5 py-1 rounded-full">Grant</span>
                  )}
                </button>
                <button 
                  onClick={requestPermissions}
                  className="w-full flex items-center justify-between p-3 bg-slate-800 hover:bg-slate-700/80 rounded-xl transition-all border border-slate-700/50"
                >
                  <span className="text-xs font-medium text-slate-300">Microphone Access</span>
                  {micPermission ? (
                    <span className="text-xs text-emerald-400 font-semibold bg-emerald-500/10 px-2.5 py-1 rounded-full">Granted</span>
                  ) : (
                    <span className="text-xs text-indigo-400 font-semibold bg-indigo-500/10 px-2.5 py-1 rounded-full">Grant</span>
                  )}
                </button>
              </div>
            </div>

            <div>
              <h2 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                <Download className="w-4 h-4 text-purple-400" />
                2. AI Models Setup
              </h2>
              {isDownloading ? (
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400">Whisper Tiny STT ({downloadProgress.whisper}%)</span>
                    </div>
                    <div className="w-full bg-slate-850 h-2 rounded-full overflow-hidden">
                      <div 
                        className="bg-indigo-500 h-full transition-all duration-300"
                        style={{ width: `${downloadProgress.whisper}%` }}
                      ></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400">Qwen 1.7B LLM ({downloadProgress.llm}%)</span>
                    </div>
                    <div className="w-full bg-slate-850 h-2 rounded-full overflow-hidden">
                      <div 
                        className="bg-purple-500 h-full transition-all duration-300"
                        style={{ width: `${downloadProgress.llm}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              ) : (
                <button
                  disabled={!screenPermission || !micPermission}
                  onClick={startDownload}
                  className="w-full py-3 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl text-sm font-semibold text-white shadow-lg shadow-indigo-500/20 transition-all"
                >
                  Download Models (~1.2 GB)
                </button>
              )}
            </div>
          </div>
          
          <button 
            onClick={() => setIsOnboarded(true)} 
            className="text-xs text-slate-500 hover:text-slate-400 underline transition-colors"
          >
            Skip onboarding (use existing files)
          </button>
        </div>
      ) : (
        /* MAIN APPLICATION VIEW */
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <header className="h-16 border-b border-slate-900 px-6 flex items-center justify-between bg-slate-950/80 backdrop-blur-md sticky top-0 z-30">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-tr from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center">
                <Sliders className="w-4 h-4 text-white" />
              </div>
              <span className="font-bold text-lg bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
                Glance
              </span>
              <div className="h-4 w-px bg-slate-800 mx-2" />
              <div className="flex items-center gap-2 bg-slate-900 px-3 py-1 rounded-full border border-slate-800">
                <span className={`w-2 h-2 rounded-full ${
                  status === "Listening" ? "bg-emerald-500 animate-ping" : 
                  status === "Inferring" ? "bg-amber-500 animate-pulse" : "bg-slate-500"
                }`} />
                <span className="text-xs font-medium text-slate-400">{status}</span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button 
                onClick={handleStartCapture}
                className="p-2.5 bg-indigo-600 hover:bg-indigo-500 rounded-xl transition-all shadow-md shadow-indigo-600/10 flex items-center gap-1.5 text-xs font-semibold"
              >
                <Play className="w-3.5 h-3.5" /> Start
              </button>
              <button 
                onClick={handleStopCapture}
                className="p-2.5 bg-slate-900 hover:bg-slate-850 rounded-xl transition-all border border-slate-800 flex items-center gap-1.5 text-xs font-semibold"
              >
                <Square className="w-3.5 h-3.5 text-slate-400" /> Stop
              </button>
              <button 
                onClick={handleRestartCapture}
                className="p-2.5 bg-slate-900 hover:bg-slate-850 rounded-xl transition-all border border-slate-800"
                title="Restart Capture"
              >
                <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
              </button>
              <button 
                onClick={() => setIsSettingsOpen(true)}
                className="p-2.5 bg-slate-900 hover:bg-slate-850 rounded-xl transition-all border border-slate-800 ml-2"
              >
                <Settings className="w-3.5 h-3.5 text-slate-400" />
              </button>
            </div>
          </header>

          {/* Split Dashboard */}
          <main className="flex-1 grid grid-cols-2 divide-x divide-slate-900 overflow-hidden h-[calc(100vh-4rem)]">
            
            {/* Left Column: Real-time Transcript */}
            <div className="flex flex-col h-full bg-slate-950/40 overflow-y-auto p-6 space-y-6">
              <div className="flex items-center justify-between border-b border-slate-900 pb-3">
                <h3 className="text-sm font-semibold text-slate-400 flex items-center gap-2">
                  <FileText className="w-4 h-4 text-indigo-400" /> Real-time Transcript
                </h3>
                <span className="text-[10px] text-slate-500 font-mono">16kHz mono</span>
              </div>

              <div className="space-y-4">
                {blocks.map((block) => (
                  <div 
                    key={block.id} 
                    className={`p-4 rounded-2xl border transition-all ${
                      block.isQuestion 
                        ? "bg-slate-900/50 border-slate-800/80 shadow-sm" 
                        : "bg-slate-950 border-transparent text-slate-400"
                    }`}
                  >
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-[10px] text-slate-500 font-mono">{block.timestamp}</span>
                      {block.isQuestion && (
                        <span className="text-[10px] text-indigo-400 font-bold bg-indigo-500/10 px-2 py-0.5 rounded">Question</span>
                      )}
                    </div>
                    <p className="text-sm leading-relaxed">{block.text}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Right Column: AI Answers */}
            <div className="flex flex-col h-full bg-slate-950/20 overflow-y-auto p-6 space-y-6">
              <div className="flex items-center justify-between border-b border-slate-900 pb-3">
                <h3 className="text-sm font-semibold text-slate-400 flex items-center gap-2">
                  <HelpCircle className="w-4 h-4 text-purple-400" /> Instant LLM Answers
                </h3>
                <span className="text-[10px] text-slate-500 font-mono">Qwen GPU</span>
              </div>

              <div className="space-y-4">
                {blocks.filter(b => b.isQuestion).map((block) => (
                  <div key={`ans-${block.id}`} className="bg-gradient-to-br from-indigo-950/20 to-purple-950/20 border border-indigo-900/30 rounded-2xl p-5 space-y-3">
                    <div className="text-xs text-indigo-300 font-semibold border-b border-indigo-900/30 pb-2">
                      Q: "{block.text}"
                    </div>
                    <p className="text-sm leading-relaxed text-purple-200">
                      {block.answer || <span className="text-slate-500 italic animate-pulse">Generating answer...</span>}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </main>
        </div>
      )}

      {/* Settings Modal */}
      {isSettingsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm">
          <div className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-2xl space-y-5 animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-slate-200">System Pre-Prompt Settings</h3>
              <button 
                onClick={() => setIsSettingsOpen(false)}
                className="p-1 hover:bg-slate-800 rounded-lg text-slate-400 transition-colors"
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
                className="w-full text-sm bg-slate-950 border border-slate-800 rounded-xl p-3 text-slate-300 focus:outline-none focus:border-indigo-500 transition-colors resize-none"
                placeholder="Enter system prompt guidelines..."
              />
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <button 
                onClick={() => setIsSettingsOpen(false)}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-xl text-xs font-semibold text-white transition-all shadow-lg shadow-indigo-600/10"
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
