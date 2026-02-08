import { useCallback, useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Navbar from "./components/Navbar";
import AudioInputCard from "./components/AudioInputCard";
import LiveRecorder from "./components/LiveRecorder";
import AnalysisPanel from "./components/AnalysisPanel";
import Stepper from "./components/Stepper";
import ModelStatusCard from "./components/ModelStatusCard";

const API_BASE = import.meta.env.VITE_API_URL || "/api";

const DEFAULT_METADATA = {
  duration: "--",
  format: "--",
  sampleRate: "--"
};

const LANGUAGE_MAP = {
  en: "English",
  es: "Spanish",
  fr: "French",
  de: "German",
  it: "Italian",
  pt: "Portuguese",
  ru: "Russian",
  ja: "Japanese",
  zh: "Chinese",
  ar: "Arabic",
  hi: "Hindi",
  te: "Telugu",
  ta: "Tamil",
  kn: "Kannada",
  ml: "Malayalam"
};

export default function App() {
  const [theme, setTheme] = useState("dark");
  const [online, setOnline] = useState(false);
  const [inputMode, setInputMode] = useState("upload"); // "upload" or "record"
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [buffer, setBuffer] = useState(null);
  const [metadata, setMetadata] = useState(DEFAULT_METADATA);
  const [targetLang, setTargetLang] = useState("en");
  const [targetEmotion, setTargetEmotion] = useState("neutral");
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [segments, setSegments] = useState([]);
  const [emotionScores, setEmotionScores] = useState({});
  const [result, setResult] = useState(null);

  useEffect(() => {
    const stored = localStorage.getItem("epmssts-theme");
    if (stored) {
      setTheme(stored);
      document.documentElement.classList.toggle("dark", stored === "dark");
    }
  }, []);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        console.log('Checking backend health at:', `${API_BASE}/health`);
        const res = await fetch(`${API_BASE}/health`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
          mode: 'cors'
        });
        console.log('Health check response:', res.status, res.ok, res.statusText);
        
        if (res.ok) {
          const data = await res.json();
          console.log('Health data:', data);
          setOnline(true);
        } else {
          console.error('Health check returned:', res.status);
          setOnline(false);
        }
      } catch (error) {
        console.error('Health check failed:', error.message);
        setOnline(false);
      }
    };
    
    // Check immediately
    checkHealth();
    
    // Then check every 5 seconds instead of 15 for faster detection
    const timer = setInterval(checkHealth, 5000);
    return () => clearInterval(timer);
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("epmssts-theme", next);
    document.documentElement.classList.toggle("dark", next === "dark");
  };

  const decodeAudio = useCallback(async (audioFile) => {
    try {
      const arrayBuffer = await audioFile.arrayBuffer();
      const context = new AudioContext();
      const audioBuffer = await context.decodeAudioData(arrayBuffer.slice(0));
      setBuffer(audioBuffer);
      setMetadata({
        duration: audioBuffer.duration.toFixed(2),
        format: audioFile.type.split("/")[1]?.toUpperCase() || "Audio",
        sampleRate: audioBuffer.sampleRate
      });
    } catch (error) {
      console.warn("Audio decode failed; continuing without waveform metadata.", error);
      setBuffer(null);
      setMetadata({
        duration: "--",
        format: audioFile.type.split("/")[1]?.toUpperCase() || "Audio",
        sampleRate: "--"
      });
    }
  }, []);

  const handleFileSelect = async (audioFile) => {
    setFile(audioFile);
    setError("");
    setResult(null);
    setSegments([]);
    setEmotionScores({});
    const url = URL.createObjectURL(audioFile);
    setAudioUrl(url);
    await decodeAudio(audioFile);
  };

  const convertToWav = async (audioFile) => {
    try {
      const arrayBuffer = await audioFile.arrayBuffer();
      const audioContext = new AudioContext();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      // Convert to WAV format
      const numberOfChannels = audioBuffer.numberOfChannels;
      const length = audioBuffer.length * numberOfChannels * 2;
      const buffer = new ArrayBuffer(44 + length);
      const view = new DataView(buffer);
      
      // WAV header
      const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(offset + i, string.charCodeAt(i));
        }
      };
      
      writeString(0, 'RIFF');
      view.setUint32(4, 36 + length, true);
      writeString(8, 'WAVE');
      writeString(12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, numberOfChannels, true);
      view.setUint32(24, audioBuffer.sampleRate, true);
      view.setUint32(28, audioBuffer.sampleRate * numberOfChannels * 2, true);
      view.setUint16(32, numberOfChannels * 2, true);
      view.setUint16(34, 16, true);
      writeString(36, 'data');
      view.setUint32(40, length, true);
      
      // Write audio data
      const offset = 44;
      const channelData = [];
      for (let i = 0; i < numberOfChannels; i++) {
        channelData.push(audioBuffer.getChannelData(i));
      }
      
      let pos = 0;
      for (let i = 0; i < audioBuffer.length; i++) {
        for (let channel = 0; channel < numberOfChannels; channel++) {
          const sample = Math.max(-1, Math.min(1, channelData[channel][i]));
          view.setInt16(offset + pos, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
          pos += 2;
        }
      }
      
      return new Blob([buffer], { type: "audio/wav" });
    } catch (error) {
      console.error('Error converting audio:', error);
      throw error;
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    
    console.log('File details:', {
      name: file.name,
      type: file.type,
      size: file.size
    });
    
    setLoading(true);
    setError("");
    setResult(null);
    setSegments([]);
    setEmotionScores({});
    setCurrentStep(0);

    try {
      // Convert to WAV if it's WebM format
      let audioFile = file;
      if (file.type === "audio/webm" || file.name.endsWith(".webm")) {
        console.log("Converting WebM to WAV...");
        try {
          const wavBlob = await convertToWav(file);
          audioFile = new File([wavBlob], file.name.replace(".webm", ".wav"), { type: "audio/wav" });
          console.log("Converted to WAV:", audioFile.type);
        } catch (error) {
          setError("Live audio conversion failed. Please try again or upload a WAV/MP3 file.");
          setLoading(false);
          return;
        }
      } else if (!file.type || !file.type.startsWith("audio/")) {
        // Ensure proper MIME type
        audioFile = new File([file], file.name, { type: "audio/wav" });
      }
      
      console.log('Sending file with type:', audioFile.type);
      
      const sttData = new FormData();
      sttData.append("file", audioFile);
      
      const sttRes = await fetch(`${API_BASE}/stt/transcribe`, {
        method: "POST",
        body: sttData
      });
      if (!sttRes.ok) {
        const errorText = await sttRes.text();
        console.error('STT Error:', errorText);
        throw new Error(`Transcription failed: ${errorText}`);
      }
      const sttJson = await sttRes.json();
      setSegments(sttJson.segments || []);
      setCurrentStep(1);

      const emotionData = new FormData();
      emotionData.append("file", audioFile);
      const emotionRes = await fetch(`${API_BASE}/emotion/detect`, {
        method: "POST",
        body: emotionData
      });
      if (!emotionRes.ok) {
        const errorText = await emotionRes.text();
        console.error('Emotion Error:', errorText);
        throw new Error(`Emotion detection failed: ${errorText}`);
      }
      const emotionJson = await emotionRes.json();
      setEmotionScores(emotionJson.scores || {});
      setCurrentStep(2);

      const dialectRes = await fetch(`${API_BASE}/dialect/detect?transcript=${encodeURIComponent(sttJson.text)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" }
      });
      if (!dialectRes.ok) throw new Error("Dialect detection failed.");
      const dialectJson = await dialectRes.json();
      setCurrentStep(3);

      const detectedLang = sttJson.language;
      const sourceLang = ["en", "hi", "te"].includes(detectedLang) ? detectedLang : "en";

      const translationRes = await fetch(`${API_BASE}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: sttJson.text,
          source_lang: sourceLang,
          target_lang: targetLang
        })
      });
      if (!translationRes.ok) throw new Error("Translation failed.");
      const translationJson = await translationRes.json();
      setCurrentStep(4);

      let outputAudioUrl = "";
      try {
        const ttsRes = await fetch(`${API_BASE}/tts/synthesize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: translationJson.translated_text,
            language: targetLang,
            emotion: targetEmotion
          })
        });
        if (ttsRes.ok) {
          const blob = await ttsRes.blob();
          outputAudioUrl = URL.createObjectURL(blob);
        } else {
          const errorText = await ttsRes.text();
          console.warn('TTS warning (non-critical):', errorText);
        }
      } catch (err) {
        console.warn('TTS error (non-critical, audio synthesis may not be available):', err);
      }

      setCurrentStep(5);

      setResult({
        transcript: sttJson.text,
        detected_language: LANGUAGE_MAP[detectedLang] || detectedLang,
        detected_emotion: emotionJson.emotion,
        detected_dialect: dialectJson.dialect,
        translated_text: translationJson.translated_text,
        target_language: LANGUAGE_MAP[targetLang] || targetLang,
        target_emotion: targetEmotion,
        output_audio_url: outputAudioUrl,
        confidence: emotionJson.confidence
      });
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const statusLabel = useMemo(() => {
    if (!loading) return "Ready";
    const labels = [
      "Speech Recognition",
      "Emotion Analysis",
      "Dialect Detection",
      "Translation",
      "Speech Synthesis"
    ];
    return labels[currentStep] || "Processing";
  }, [loading, currentStep]);

  return (
    <div className="min-h-screen flex flex-col app-shell">
      <Navbar online={online} onToggleTheme={toggleTheme} theme={theme} />

      <main className="flex-1 px-6 py-10">
        <div className="max-w-6xl mx-auto space-y-8">
          <section className="grid lg:grid-cols-[1.2fr_0.8fr] gap-6">
            <div className="space-y-4">
              <div className="flex flex-wrap gap-3">
                <span className="badge-soft">Enterprise AI</span>
                <span className="badge-soft">Real-time Processing</span>
                <span className="badge-soft">Emotion Fidelity</span>
              </div>
              <h1 className="text-3xl lg:text-4xl font-semibold leading-tight">
                Emotion-Preserving Multilingual Speech Translation System
              </h1>
              <p className="text-white/70 text-sm leading-relaxed max-w-2xl">
                EPMSSTS delivers end-to-end speech understanding, accurate emotion detection, 
                multilingual translation, and emotion-preserving speech synthesis powered by 5 AI models.
              </p>
              <div className="flex flex-wrap gap-4 pt-2">
                <div className="space-y-1">
                  <p className="text-white/50 text-xs">Supported Languages</p>
                  <p className="text-white/90 font-semibold">English, Telugu, Hindi + 197 more</p>
                </div>
              </div>
            </div>
            <ModelStatusCard online={online} />
          </section>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="gradient-border"
          >
            <div className="card-surface px-6 py-5 flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold">Intelligence Pipeline</h2>
                  <p className="text-sm text-white/60">Speech Understanding → Translation → Emotion-Aware Synthesis</p>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${loading ? "bg-amber-400 animate-pulse" : "bg-emerald-400"}`} />
                  <span className="text-xs uppercase tracking-wide text-white/50">{statusLabel}</span>
                </div>
              </div>
              <Stepper currentStep={currentStep} />
            </div>
          </motion.div>

          {error && (
            <div className="card-surface p-4 border border-rose-500/30 bg-rose-500/10 text-rose-100 text-sm">
              {error}
              <button
                onClick={() => setError("")}
                className="ml-3 text-xs underline"
              >
                Dismiss
              </button>
            </div>
          )}

          <div className="flex items-center gap-3 mb-4">
            <button
              onClick={() => setInputMode("upload")}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition ${
                inputMode === "upload"
                  ? "bg-accent-500/20 text-accent-400 border border-accent-500/30"
                  : "bg-white/5 text-white/60 hover:bg-white/10"
              }`}
            >
              📁 Upload Audio
            </button>
            <button
              onClick={() => setInputMode("record")}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition ${
                inputMode === "record"
                  ? "bg-accent-500/20 text-accent-400 border border-accent-500/30"
                  : "bg-white/5 text-white/60 hover:bg-white/10"
              }`}
            >
              🎙️ Live Recording
            </button>
          </div>

          <div className="grid lg:grid-cols-[360px_1fr] gap-6">
            <AnimatePresence mode="wait">
              {inputMode === "upload" ? (
                <AudioInputCard
                  key="upload"
                  file={file}
                  audioUrl={audioUrl}
                  buffer={buffer}
                  metadata={metadata}
                  onFileSelect={handleFileSelect}
                  onAnalyze={handleAnalyze}
                  isLoading={loading}
                  targetLang={targetLang}
                  targetEmotion={targetEmotion}
                  setTargetLang={setTargetLang}
                  setTargetEmotion={setTargetEmotion}
                />
              ) : (
                <LiveRecorder
                  key="record"
                  onRecordingComplete={handleFileSelect}
                  isProcessing={loading}
                  targetLang={targetLang}
                  targetEmotion={targetEmotion}
                  setTargetLang={setTargetLang}
                  setTargetEmotion={setTargetEmotion}
                  onAnalyze={handleAnalyze}
                />
              )}
            </AnimatePresence>

            <AnalysisPanel result={result} segments={segments} emotionScores={emotionScores} />
          </div>
        </div>
      </main>

      <footer className="px-6 py-6 text-xs text-white/50 border-t border-white/10">
        <div className="max-w-6xl mx-auto flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-4">
            <span>EPMSSTS © 2026</span>
            <span className="w-1 h-1 rounded-full bg-white/30" />
            <span>Emotion-Preserving Multilingual Speech-to-Speech Translation System</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="badge bg-white/5">5 AI Models Active</span>
            <span className="badge bg-white/5">200+ Languages</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
