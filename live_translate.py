

from __future__ import annotations
import os, sys, re, time, json, queue, argparse, threading, socket, glob
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Any

# ---------- Environment speedups (set BEFORE heavy imports) ----------
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))
os.environ.setdefault("MKL_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Windows: add CUDA bin dirs if present so DLLs load when you switch to --device cuda
if sys.platform.startswith("win"):
    for bin_dir in glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin"):
        try:
            os.add_dll_directory(bin_dir)
        except Exception:
            pass

import numpy as np
import sounddevice as sd
import webrtcvad
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# --- Optional: global hotkeys (F1/F2/ESC) ---
try:
    import keyboard  # pip install keyboard
except Exception:
    keyboard = None  # hotkeys will be disabled if unavailable

console = Console()

# --------------------- Language utilities ---------------------
NLLB_LANG = {"en": "eng_Latn", "ru": "rus_Cyrl", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn"}
def nllb_code(lang: str) -> str:
    return NLLB_LANG.get(lang.lower(), "eng_Latn")

# --------------------- Config ---------------------
@dataclass(slots=True)
class AudioConfig:
    samplerate: int = 16000
    channels: int = 1
    block_ms: int = 20  # must be 10, 20, or 30 for WebRTC VAD
    vad_aggressiveness: int = 3
    max_silence_ms: int = 320
    device: Optional[int] = None
    max_segment_ms: int = 8000  # hard cap per utterance to prevent repetition bursts

@dataclass(slots=True)
class WhisperConfig:
    model_size: str = "small"    # tiny/base/small/medium/large-v3
    device: str = "auto"         # cuda|cpu|auto
    compute_type: Optional[str] = None  # float16|int8|int8_float16|float32
    beam_size: int = 1
    vad_filter: bool = False
    language: Optional[str] = "en"  # force English ASR by default
    initial_prompt: Optional[str] = (
        "Church sermon, biblical terms, Matthew, Mark, Luke, John, Paul, Holy Spirit, Lord's Prayer, Hallelujah, Gospel, altar, anointing"
    )

@dataclass(slots=True)
class MTConfig:
    model_name: str = "Helsinki-NLP/opus-mt-en-ru"  # fast, accurate on CPU
    src_lang: str = "eng_Latn"
    tgt_lang: str = "rus_Cyrl"
    max_new_tokens: int = 128

@dataclass(slots=True)
class PiperConfig:
    enabled: bool = False
    exe_path: Optional[str] = None
    voice_path: Optional[str] = None
    engine: str = "silero"            # "silero" or "piper"
    tts_voice: str = "aidar"          # silero voices: aidar/eugene/nikolay/baya/xenia
    female_voice: str = "xenia"       # preferred female voice for toggling (silero only)

# --------------------- LAN Audio Server Config ---------------------
@dataclass(slots=True)
class LANConfig:
    host: str = "0.0.0.0"   # listen on all interfaces
    port: int = 8765        # http://<server-ip>:8765
    enable: bool = True

# --------------------- Helpers ---------------------
_LATIN_RATIO_MIN = 0.85
def mostly_latin(s: str) -> bool:
    if not s:
        return False
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return False
    latin = sum('A' <= ch <= 'Z' or 'a' <= ch <= 'z' for ch in letters)
    return (latin / max(1, len(letters))) >= _LATIN_RATIO_MIN

# Robust scripture reference patterns
REF_BOOKS = {
    r"Matt(?:hew)?\.?": "Matthew",
    r"Mark\.?": "Mark",
    r"Luke\.?": "Luke",
    r"John\.?": "John",
    r"Rom(?:ans)?\.?": "Romans",
}
_ref_patterns = [
    (re.compile(fr"\b(?:{pat})\s*(?P<ch>\d+)\s*[:\s]\s*(?P<vs>\d+)\b", re.I), book)
    for pat, book in REF_BOOKS.items()
]
def fix_refs(s: str) -> str:
    out = s
    for pat, book in _ref_patterns:
        out = pat.sub(lambda m: f"{book} {m.group('ch')}:{m.group('vs')}", out)
    return out

# --------- ASR Guard helpers (anti-hallucination) ----------
@dataclass(slots=True)
class AsrGuard:
    enable: bool = True
    min_avg_logprob: float = -0.65
    max_compression_ratio: float = 2.2
    max_no_speech_prob: float = 0.6
    max_repeat_run: int = 6      # drop if any single word repeats this many times
    dedupe_repeats: bool = True  # collapse "very very very" -> "very"

def collapse_repeats(text: str) -> str:
    return re.sub(r"\b(\w+)(?:\s+\1\b){1,}\s*", r"\1 ", text, flags=re.IGNORECASE).strip()

def has_long_repeat_run(text: str, n: int) -> bool:
    run, last = 1, None
    for w in re.findall(r"[A-Za-z']+", text):
        wl = w.lower()
        run = run + 1 if wl == last else 1
        last = wl
        if run >= n:
            return True
    return False

# ---------- Ultra-fast generic noise/repetition gate ----------
from collections import Counter
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # Unicode letters across scripts

def _tok_fast(s: str) -> List[str]:
    return _WORD_RE.findall(s.lower())

def _uniq_ratio(toks: List[str]) -> float:
    n = len(toks)
    return (len(set(toks)) / n) if n else 0.0

def _longest_run(toks: List[str]) -> int:
    if not toks: return 0
    longest = run = 1
    prev = toks[0]
    for t in toks[1:]:
        if t == prev:
            run += 1
            if run > longest: longest = run
        else:
            run = 1
            prev = t
    return longest

def _max_ngram_share(toks: List[str], n: int) -> float:
    m = len(toks) - n + 1
    if m <= 1: return 0.0
    counts = Counter(tuple(toks[i:i+n]) for i in range(m))
    return max(counts.values()) / m

@dataclass(slots=True)
class FastGateCfg:
    enable: bool = True
    max_words: int = 80            # drop very long rambles
    min_unique_ratio: float = 0.46 # if too many repeats, drop
    max_repeat_run: int = 6        # "word word word word word word" -> drop
    max_bigram_share: float = 0.40 # one bigram dominates stream
    max_trigram_share: float = 0.32

class FastNoiseGate:
    def __init__(self, cfg: FastGateCfg = FastGateCfg()):
        self.cfg = cfg

    def drop_src(self, text: str) -> bool:
        if not self.cfg.enable or not text: return False
        toks = _tok_fast(text)
        n = len(toks)
        if n < 3:      return True
        if n > self.cfg.max_words: return True
        if _uniq_ratio(toks) < self.cfg.min_unique_ratio: return True
        if _longest_run(toks) >= self.cfg.max_repeat_run: return True
        if n >= 8 and _max_ngram_share(toks, 2) > self.cfg.max_bigram_share: return True
        if n >= 10 and _max_ngram_share(toks, 3) > self.cfg.max_trigram_share: return True
        return False

    def drop_tgt(self, text: str) -> bool:
        # Looser target gate (Unicode-aware)
        if not self.cfg.enable or not text: return False
        toks = _tok_fast(text)
        if not toks: return True
        if _longest_run(toks) >= self.cfg.max_repeat_run: return True
        if len(toks) >= 8 and _max_ngram_share(toks, 2) > self.cfg.max_bigram_share: return True
        return False

# --------------------- Audio + VAD capture ---------------------
class VADSegmenter:
    def __init__(self, cfg: AudioConfig):
        if cfg.block_ms not in (10, 20, 30):
            raise ValueError("block_ms must be 10, 20, or 30 for WebRTC VAD")
        self.cfg = cfg
        self.vad = webrtcvad.Vad(cfg.vad_aggressiveness)
        self.block_size = int(cfg.samplerate * cfg.block_ms / 1000)
        self.silence_blocks = max(1, cfg.max_silence_ms // cfg.block_ms)
        self.max_blocks = max(1, cfg.max_segment_ms // cfg.block_ms)
        self.frames: List[bytes] = []
        self.silent_count = 0

    def _emit(self) -> Optional[np.ndarray]:
        if not self.frames:
            return None
        chunk = b"".join(self.frames)
        self.frames = []
        self.silent_count = 0
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        return audio

    def process_block(self, pcm16: bytes) -> Optional[np.ndarray]:
        is_speech = self.vad.is_speech(pcm16, sample_rate=self.cfg.samplerate)
        if is_speech:
            self.frames.append(pcm16)
            self.silent_count = 0
            if len(self.frames) >= self.max_blocks:  # force-cut very long chunks
                return self._emit()
            return None
        else:
            if self.frames:
                self.silent_count += 1
                self.frames.append(pcm16)
                if self.silent_count >= self.silence_blocks:
                    return self._emit()
            return None

# --------------------- ASR (faster-whisper) ---------------------
class Transcriber:
    def __init__(self, cfg: WhisperConfig):
        device = cfg.device
        if device == "auto":
            device = "cuda" if _has_cuda() else "cpu"
        compute_type = cfg.compute_type or ("float16" if device == "cuda" else "int8")
        console.print(f"[cyan]Loading Whisper '{cfg.model_size}' on {device} ({compute_type})...[/cyan]")
        self.model = WhisperModel(
            cfg.model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=max(1, (os.cpu_count() or 4) - 1)
        )
        self.cfg = cfg

    def transcribe(self, audio: np.ndarray, samplerate: int) -> Tuple[str, float, dict]:
        start = time.time()
        segments, info = self.model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=max(2, self.cfg.beam_size),     # small quality boost
            vad_filter=self.cfg.vad_filter,
            word_timestamps=False,
            condition_on_previous_text=False,
            temperature=0.0,
            compression_ratio_threshold=2.2,         # tighter anti-repeat
            log_prob_threshold=-0.65,                 # drop low-confidence
            no_speech_threshold=0.6,
            initial_prompt=self.cfg.initial_prompt,
        )
        text = "".join(seg.text for seg in segments).strip()
        avg_lp = float(np.mean([getattr(seg, "avg_logprob", -10.0) for seg in segments])) if segments else -10.0
        cr = float(np.mean([getattr(seg, "compression_ratio", 0.0) for seg in segments])) if segments else 0.0
        ns = float(np.mean([getattr(seg, "no_speech_prob", 1.0) for seg in segments])) if segments else 1.0
        stats = {"avg_logprob": avg_lp, "compression_ratio": cr, "no_speech_prob": ns}
        return text, (time.time() - start), stats

def _has_cuda() -> bool:
    try:
        import ctranslate2  # used by faster-whisper
        return getattr(ctranslate2, "get_cuda_device_count", lambda: 0)() > 0
    except Exception:
        return bool(os.environ.get("CUDA_VISIBLE_DEVICES"))

# --------------------- Translator (Marian or NLLB) ---------------------
class Translator:
    def __init__(self, cfg: MTConfig):
        self.cfg = cfg
        console.print(f"[cyan]Loading translator '{cfg.model_name}'...[/cyan]")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name, use_safetensors=True)

        if "opus-mt-" in cfg.model_name:
            task = "translation_en_to_ru"
            self.pipe = pipeline(
                task,
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=min(128, self.cfg.max_new_tokens),
                truncation=True,
                num_beams=1,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
            )
        else:
            self.pipe = pipeline(
                "translation",
                model=self.model,
                tokenizer=self.tokenizer,
                src_lang=cfg.src_lang,
                tgt_lang=cfg.tgt_lang,
                max_new_tokens=self.cfg.max_new_tokens,
                truncation=True,
            )

    def translate(self, text: str) -> str:
        if not text:
            return ""
        return self.pipe(text)[0]["translation_text"]

# --------------------- Silero TTS (CPU, offline) ---------------------
class SileroTTS:
    """
    CPU-friendly Russian TTS using Silero (offline).
    Voices: 'aidar' (male), 'eugene' (male), 'nikolay' (male), 'baya' (female), 'xenia' (female)
    """
    def __init__(self, speaker: str = "aidar", sample_rate: int = 24000):
        import torch
        self.torch = torch
        self.sample_rate = int(sample_rate)
        self.speaker = speaker
        self.model, _ = torch.hub.load(
            'snakers4/silero-models', 'silero_tts',
            language='ru', speaker='v3_1_ru', trust_repo=True
        )

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        if not text or not text.strip():
            return np.zeros(0, dtype=np.float32), self.sample_rate
        with self.torch.no_grad():
            audio = self.model.apply_tts(text=text, speaker=self.speaker, sample_rate=self.sample_rate)
        return np.array(audio, dtype=np.float32), self.sample_rate

# --------------------- Optional Piper TTS ---------------------
class PiperTTS:
    def __init__(self, cfg: PiperConfig):
        self.enabled = bool(cfg.enabled and cfg.exe_path and cfg.voice_path
                            and Path(cfg.exe_path).exists() and Path(cfg.voice_path).exists())
        self.exe = cfg.exe_path
        self.voice = cfg.voice_path

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        if not (self.enabled and text and text.strip()):
            return np.zeros(0, dtype=np.float32), 22050
        import subprocess, tempfile, soundfile as sf, os
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
                wav_path = wf.name
            cmd = [self.exe, "-m", self.voice, "-f", wav_path, "-q"]
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                p.communicate(input=text.encode("utf-8"), timeout=30)
            except subprocess.TimeoutExpired:
                p.kill(); p.communicate()
                raise RuntimeError("piper timed out")
            if p.returncode != 0:
                raise RuntimeError("piper failed")
            data, sr = sf.read(wav_path, dtype="float32")
            return data.astype(np.float32, copy=False), int(sr)
        finally:
            if wav_path and os.path.exists(wav_path):
                try: os.remove(wav_path)
                except Exception: pass

# --------------------- Glossary ---------------------
class Glossary:
    def __init__(self, path: Path, tgt_lang: str):
        self.map = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("target_lang") in (tgt_lang, None):
                    self.map = data.get("replacements", {})
            except Exception as e:
                console.print(f"[yellow]Failed to load glossary ({e}). Continuing without it.[/yellow]")

    def apply(self, text: str) -> str:
        if not self.map or not text:
            return text
        out = text
        for k in sorted(self.map, key=len, reverse=True):
            v = self.map[k]
            out = re.sub(rf"\b{re.escape(k)}\b", v, out)
        return out

# --------------------- Faith normalizer (God-name preference) ---------------------
@dataclass(slots=True)
class FaithNormConfig:
    normalize_allah_to_bog: bool = True
    bog_case: str = "lower"  # "lower" or "upper"

class FaithNormalizer:
    _allah_map_lower = {
        "аллах":   "бог",
        "аллаха":  "бога",
        "аллаху":  "богу",
        "аллахом": "богом",
        "аллахе":  "боге",
    }
    def __init__(self, cfg: FaithNormConfig):
        self.cfg = cfg
        self._allah_map_target = (
            {k: v.capitalize() for k, v in self._allah_map_lower.items()}
            if cfg.bog_case == "upper" else dict(self._allah_map_lower)
        )
        self._allah_pat = re.compile(r"\b(Аллах(?:[ауеом])?)\b", re.IGNORECASE)

    def _allah_repl(self, m: re.Match) -> str:
        src = m.group(0)
        dst = self._allah_map_target.get(src.lower())
        return dst if dst else src

    def apply(self, text: str) -> str:
        if not text:
            return text
        out = text
        out = self._allah_pat.sub(self._allah_repl, out)
        if self.cfg.bog_case == "lower":
            out = re.sub(r"\b(Бог|Бога|Богу|Богом|Боге)\b", lambda m: m.group(0).lower(), out)
        return out

# --------------------- LAN audio+transcript server (WebSocket) ---------------------
class LANAudioServer:
    """
    Broadcasts audio + transcripts over LAN.
    Uses a hidden <audio> sink fed by MediaStreamAudioDestinationNode so playback continues with screen off.
    """
    HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LAN TTS</title>
  <style>
    :root { --bg:#fafafa; --fg:#222; --muted:#727272; --card:#fff; --accent:#007acc; --border:#dadada; --radius:12px; --shadow-1:rgba(0,0,0,.06); --shadow-2:rgba(0,0,0,.12); --font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Oxygen,Ubuntu,Cantarell,"Open Sans","Helvetica Neue",sans-serif; }
    @media (prefers-color-scheme: dark) { :root { --bg:#0e0f12; --fg:#e9e9ea; --muted:#a0a0ad; --card:#14161a; --border:#282b31; --accent:#4aa3ff; --shadow-1:rgba(0,0,0,.4); --shadow-2:rgba(0,0,0,.6); } }
    *,*::before,*::after { box-sizing:border-box }
    html,body { height:100%; margin:0 }
    body { background:var(--bg); color:var(--fg); font-family:var(--font); font-size:16px; line-height:1.5; -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale }
    header { position:sticky; top:0; z-index:10; background:var(--card); border-bottom:1px solid var(--border); box-shadow:0 2px 6px var(--shadow-1); padding:14px 20px; display:flex; flex-wrap:wrap; gap:12px; align-items:center; justify-content:space-between }
    .row { display:flex; flex-wrap:wrap; gap:12px; align-items:center }
    button { cursor:pointer; border:0; border-radius:var(--radius); background:var(--accent); color:#fff; padding:12px 18px; font-weight:600; font-size:1rem; box-shadow:0 2px 4px var(--shadow-2); transition:transform .1s ease, box-shadow .2s ease, background-color .25s ease; user-select:none; -webkit-tap-highlight-color:transparent }
    button:hover:not(:disabled), button:focus-visible:not(:disabled) { background:#005ea3; box-shadow:0 4px 12px var(--shadow-2); transform:translateY(-1px) }
    button:disabled { opacity:.65; cursor:default; box-shadow:none }
    button:focus-visible { outline:2px solid var(--accent); outline-offset:2px }
    label { font-size:.95rem; color:var(--muted); display:inline-flex; align-items:center; gap:6px; cursor:pointer; user-select:none }
    input[type="checkbox"] { width:18px; height:18px }
    main { max-width:900px; margin:20px auto 40px; padding:0 16px; width:100% }
    #status { font-size:.9rem; color:var(--muted); min-width:120px; white-space:nowrap }
    #feed { margin-top:20px; display:grid; gap:14px; width:100% }
    .card { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:14px 16px; box-shadow:0 1px 4px var(--shadow-1); word-break:break-word; overflow-wrap:anywhere; font-size:1.05rem; transition:box-shadow .2s ease }
    .card:hover { box-shadow:0 6px 16px var(--shadow-2) }
    .meta { font-size:.85rem; color:var(--muted); margin-bottom:8px }
    .kv .k { color:#444; font-weight:600 }
    .kv .v { color:var(--accent); margin-bottom:4px; font-weight:500 }
    .hidden { display:none!important }
    pre.log { background:var(--bg); border:1px solid var(--border); border-radius:10px; padding:10px; max-height:220px; overflow:auto; font-size:.95rem; margin-top:18px }
    @media (max-width:600px){ header,.row{ flex-direction:column; align-items:stretch } button,label{ width:100%; font-size:1.05rem; padding:14px 0; justify-content:center } #status{ text-align:center; min-width:auto; white-space:normal } .card{ font-size:1.1rem } }
    .corner-note{ position:fixed; bottom:calc(16px + env(safe-area-inset-bottom)); right:calc(12px + env(safe-area-inset-right)); font-size:.85rem; color:var(--muted); opacity:.85; background:transparent; padding:6px 8px; border-radius:8px; user-select:none; z-index:999; pointer-events:none }
    .corner-note a{ color:inherit; text-decoration:none; border-bottom:1px dotted currentColor; pointer-events:auto }
    .corner-note a:hover{ text-decoration:underline; opacity:1 }
    #feed::after{ content:""; display:block; height:calc(70px + env(safe-area-inset-bottom)) }
  </style>
</head>
<body>
  <header>
    <div class="row">
      <button id="btn" type="button">Join the Russian Channel</button>
      <span id="status" role="status" aria-live="polite">Idle</span>
    </div>
    <div class="row">
      <label><input type="checkbox" id="transToggle" checked> Show transcript</label>
      <label><input type="checkbox" id="awakeToggle" checked> Keep screen awake</label>
      <label><input type="checkbox" id="logToggle"> Debug log</label>
    </div>
  </header>
  <main>
    <div id="feed"></div>
    <pre id="log" class="log hidden"></pre>
  </main>

  <!-- Hidden audio sink that keeps playing when the screen is off -->
  <audio id="sink" autoplay playsinline></audio>

  <div class="corner-note" role="contentinfo" aria-label="copyright and license">
    © 2025 Nitesh Morem • Docs:
    <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener noreferrer">CC BY 4.0</a>
    • Code:
    <a href="https://opensource.org/license/mit" target="_blank" rel="noopener noreferrer">MIT</a>
  </div>

  <script>
    const $ = sel => document.querySelector(sel);
    const FEED_MAX = 50, MAGIC = 0x50434d30; // 'PCM0'
    let ctx = null, node = null, ws = null, wakeLock = null;
    let cur = null, idx = 0, playing = false, joined = false;
    let logEnabled = false, transEnabled = true;
    let reconnectTimer = null, backoff = 1000;
    const setStatus = t => { const s = $('#status'); if (s) s.textContent = t; };
    const log = m => { if (!logEnabled) return; const el = $('#log'); el.textContent += m + "\n"; el.scrollTop = el.scrollHeight; };

    async function requestWakeLock(){ if(!('wakeLock' in navigator)) { log('Wake Lock API not available'); return; }
      try{ wakeLock = await navigator.wakeLock.request('screen'); wakeLock.addEventListener('release', ()=>log('Wake lock released')); log('Wake lock active'); }catch(e){ log('Wake lock failed: '+(e?.message||e)); } }
    async function releaseWakeLock(){ try{ if(wakeLock){ await wakeLock.release(); wakeLock = null; } }catch{} }

    function setupAudio(){ if(ctx) return;
      try { ctx = new (window.AudioContext||window.webkitAudioContext)(); }
      catch { ctx = new (window.AudioContext||window.webkitAudioContext)(); }
      node = (ctx.createScriptProcessor||ctx.createJavaScriptNode)?.call(ctx, 2048, 0, 1);
      node.onaudioprocess = e => {
        const out = e.outputBuffer.getChannelData(0); let i = 0, n = out.length;
        while(i < n){
          if(!cur || idx >= cur.length){
            cur = (queue.length ? queue.shift() : null); idx = 0;
            if(!cur){ out.fill(0, i); if(playing){ playing=false; setStatus('Buffering…'); } break; }
          }
          const take = Math.min(n - i, cur.length - idx);
          out.set(cur.subarray(idx, idx + take), i);
          i += take; idx += take;
          if(!playing){ playing = true; setStatus(`Playing @ ${ctx.sampleRate} Hz…`); }
        }
      };

      // ---- Screen-lock-safe route: WebAudio -> MediaStream -> <audio id="sink">
      const dest = ctx.createMediaStreamDestination();
      node.connect(dest);
      const sinkEl = document.getElementById('sink');
      try {
        sinkEl.srcObject = dest.stream;
        sinkEl.play().catch(()=>{});
      } catch (_) {}

      // Optional: lock-screen metadata
      if ('mediaSession' in navigator) {
        try {
          navigator.mediaSession.metadata = new MediaMetadata({
            title: 'Live RU Channel',
            artist: 'Church Translation',
            album: 'Live Stream'
          });
          ['play','pause','stop','seekbackward','seekforward','previoustrack','nexttrack'].forEach(a=>{
            try { navigator.mediaSession.setActionHandler(a, ()=>{}); } catch(_) {}
          });
        } catch(_){}
      }
    }

    function teardownAudio(){ try{ node?.disconnect(); }catch{}; try{ ctx?.close(); }catch{}; node=null; ctx=null; cur=null; idx=0; playing=false; queue.length=0; }
    function resampleFloat32(data, srcRate, dstRate){ if(srcRate===dstRate) return data; const ratio = srcRate/dstRate; const newLen = Math.max(1, Math.round(data.length/ratio)); const out = new Float32Array(newLen); for(let i=0;i<newLen;i++){ const j=i*ratio, j0=j|0, frac=j-j0; const a=data[j0]||0, b=data[j0+1]??a; out[i]=a+frac*(b-a); } return out; }

    function addCard(msg){ if(!transEnabled) return; const feed = $('#feed'); const card = document.createElement('div'); card.className='card';
      card.innerHTML = `<div class="meta">ASR ${msg.asr_ms} ms • MT ${msg.mt_ms} ms • Total ${msg.total_ms} ms</div>
                        <div class="kv"><div class="k">Source</div><div class="v">${msg.src||''}</div>
                        <div class="k">Translation</div><div class="v">${msg.tgt||''}</div></div>`;
      feed.append(card); while(feed.childElementCount > FEED_MAX) feed.firstElementChild.remove();
    }

    const queue = [];
    function startWS(){ if(ws) return;
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.binaryType = 'arraybuffer';
      ws.onopen  = () => { setStatus('Connected. Waiting for audio…'); log('WebSocket connected'); backoff = 1000; };
      ws.onclose = () => { log('WebSocket disconnected'); ws = null; setStatus('Disconnected.'); if(joined) scheduleReconnect(); };
      ws.onerror = () => { log('WebSocket error'); };
      ws.onmessage = ev => {
        try{
          if(typeof ev.data === 'string'){
            const msg = JSON.parse(ev.data);
            if(msg?.type === 'trans'){ addCard(msg); return; }
            if(msg?.type === 'ctrl' && 'captions' in msg){
              transEnabled = msg.captions !== 'off';
              $('#feed').classList.toggle('hidden', !transEnabled);
              setStatus(transEnabled ? 'Connected. Waiting for audio…' : 'Captions paused by host');
              return;
            }
            return;
          }
          const dv = new DataView(ev.data);
          if(dv.getUint32(0, false) !== MAGIC){ log('Unknown binary payload'); return; }
          const sr = dv.getUint32(4, true);
          const f32 = new Float32Array(ev.data, 12);
          const data = ctx ? resampleFloat32(f32, sr, ctx.sampleRate) : f32;
          queue.push(data);
          if(!playing && ctx) setStatus(`Playing @ ${ctx.sampleRate} Hz…`);
        }catch(ex){ log('WS message error: '+ex.message); }
      };
    }
    function scheduleReconnect(){ if(reconnectTimer) return; reconnectTimer = setTimeout(()=>{ reconnectTimer=null; if(!ws && joined){ log('Reconnecting…'); startWS(); backoff = Math.min(backoff*1.6, 8000); } }, backoff); }
    function stopAll(){ joined=false; try{ ws?.close(); }catch{} ws=null; clearTimeout(reconnectTimer); reconnectTimer=null; releaseWakeLock(); teardownAudio(); setStatus('Idle'); $('#btn').textContent = 'Join the Russian Channel'; }

    $('#btn').onclick = async () => { if(!joined){ joined = true; $('#btn').textContent = 'Leave channel'; setupAudio(); startWS(); if($('#awakeToggle').checked) await requestWakeLock(); } else { stopAll(); } };
    document.addEventListener('visibilitychange', async () => { if(document.visibilityState === 'visible' && $('#awakeToggle').checked) await requestWakeLock(); });
    $('#logToggle').onchange   = e => { logEnabled = e.target.checked; $('#log').classList.toggle('hidden', !logEnabled); if(logEnabled) log('Debug log enabled'); };
    $('#transToggle').onchange = e => { transEnabled = e.target.checked; $('#feed').classList.toggle('hidden', !transEnabled); };
    $('#awakeToggle').onchange = async e => { if(e.target.checked) await requestWakeLock(); else await releaseWakeLock(); };
  </script>
</body>
</html>
"""

    def __init__(self, cfg: LANConfig):
        self.cfg = cfg
        # Unified queue: ('A', samples(np.float32), sr[int]) OR ('T', json_str)
        self._q: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=64)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def send(self, samples: np.ndarray, sample_rate: int):
        try:
            self._q.put_nowait(('A', samples.astype(np.float32, copy=False), int(sample_rate)))
        except queue.Full:
            pass

    def send_text(self, payload: dict):
        try:
            js = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
            self._q.put_nowait(('T', js))
        except queue.Full:
            pass

    def _run(self):
        import asyncio, struct
        from contextlib import suppress
        try:
            from aiohttp import web
        except Exception as e:
            console.print(f"[red]aiohttp not installed: {e}. LAN broadcast disabled.[/red]")
            return

        try:
            if sys.platform.startswith("win"):
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

        async def index(_):
            return web.Response(text=self.HTML_PAGE, content_type="text/html")

        websockets: "set[web.WebSocketResponse]" = set()

        async def ws_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            websockets.add(ws)
            try:
                async for _ in ws:
                    pass
            finally:
                websockets.discard(ws)
            return ws

        async def broadcaster():
            loop = asyncio.get_running_loop()
            while True:
                item = await loop.run_in_executor(None, self._q.get)
                kind = item[0]
                dead = []
                if kind == 'A':
                    _, samples, sr = item
                    header = struct.pack("<4sII", b"PCM0", sr, samples.shape[0])
                    payload = header + samples.tobytes(order="C")
                    for ws in list(websockets):
                        try:    await ws.send_bytes(payload)
                        except Exception: dead.append(ws)
                elif kind == 'T':
                    _, js = item
                    for ws in list(websockets):
                        try:    await ws.send_str(js)
                        except Exception: dead.append(ws)
                for d in dead:
                    websockets.discard(d)

        async def on_startup(app):
            app['broadcaster_task'] = asyncio.create_task(broadcaster())
            console.print(f"[green]LAN server listening on http://{guess_lan_ip()}:{self.cfg.port}[/green]")

        async def on_cleanup(app):
            task = app.get('broadcaster_task')
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        app = web.Application()
        app.router.add_get("/", index)
        app.router.add_get("/ws", ws_handler)
        app.on_startup.append(on_startup)
        app.on_cleanup.append(on_cleanup)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, self.cfg.host, self.cfg.port)
        try:
            loop.run_until_complete(site.start())
        except OSError as e:
            console.print(f"[red]Failed to bind {self.cfg.host}:{self.cfg.port} — {e}[/red]")
            return
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(runner.cleanup())
            loop.close()

# --------------------- Main App with concurrency ---------------------
class LiveTranslatorApp:
    def __init__(self, a_cfg: AudioConfig, w_cfg: WhisperConfig, m_cfg: MTConfig, p_cfg: PiperConfig, glossary_path: Path, lan_cfg: LANConfig):
        self.audio_cfg = a_cfg
        self.segmenter = VADSegmenter(a_cfg)
        self.transcriber = Transcriber(w_cfg)
        self.translator = Translator(m_cfg)
        self.glossary = Glossary(glossary_path, m_cfg.tgt_lang)
        self.faithnorm = FaithNormalizer(FaithNormConfig(bog_case="lower"))  # change to "upper" for Бог/Бога…

        # --- TTS selection ---
        if p_cfg.engine == "silero" and p_cfg.enabled is not False:
            self.tts = SileroTTS(speaker=p_cfg.tts_voice)
        elif p_cfg.engine == "piper" and p_cfg.enabled is not False:
            self.tts = PiperTTS(p_cfg)
        else:
            class _NoTTS:
                def synthesize(self, text: str): return (np.zeros(0, dtype=np.float32), 24000)
            self.tts = _NoTTS()

        # Hotkeys & state
        self.tts_muted = False
        self._last_esc = 0.0

        # Voice toggle (male ↔ female) for Silero
        self._male_voices = {"aidar", "eugene", "nikolay"}
        self._female_voices = {"baya", "xenia"}
        if isinstance(self.tts, SileroTTS):
            male_voice = p_cfg.tts_voice if p_cfg.tts_voice in self._male_voices else "aidar"
            female_voice = p_cfg.female_voice if p_cfg.female_voice in self._female_voices else "xenia"
            self.voice_cycle = [male_voice, female_voice]
            if getattr(self.tts, "speaker", None) == female_voice:
                self.voice_idx = 1
            else:
                self.tts.speaker = male_voice
                self.voice_idx = 0
        else:
            self.voice_cycle, self.voice_idx = [], 0

        # Queues
        self.asr_guard = AsrGuard()
        self.fast_gate = FastNoiseGate()  # <-- NEW
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.translate_q: queue.Queue[Tuple[str, float, float, float]] = queue.Queue(maxsize=8)
        self.print_q: queue.Queue[Tuple[str, str, float, float, float, float]] = queue.Queue(maxsize=16)
        self.tts_q: queue.Queue[str] = queue.Queue(maxsize=32)

        self.stop_event = threading.Event()

        # Warm-up
        _ = self.translator.translate("Hello")
        dummy = np.zeros(int(self.audio_cfg.samplerate * 0.5), dtype=np.float32)
        _t, _r, _s = self.transcriber.transcribe(dummy, self.audio_cfg.samplerate)

        # Workers
        self.mt_thread = threading.Thread(target=self._mt_worker, daemon=True)
        self.mt_thread.start()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

        # LAN
        self.lan = LANAudioServer(lan_cfg) if lan_cfg.enable else None
        self.lan_cfg = lan_cfg

    @staticmethod
    def _drain_queue(q: "queue.Queue"):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    # ---------- Hotkeys ----------
    def _toggle_mute(self):
        self.tts_muted = not self.tts_muted
        state = "muted" if self.tts_muted else "unmuted"
        console.print(f"[magenta]Broadcast: [bold]{state}[/bold].[/magenta]")
        if self.tts_muted:
            # Drop pending work so resume is fresh
            self._drain_queue(self.q)
            self._drain_queue(self.translate_q)
            self._drain_queue(self.print_q)
            self._drain_queue(self.tts_q)
            if self.lan: self.lan.send_text({"type": "ctrl", "captions": "off"})
        else:
            if self.lan: self.lan.send_text({"type": "ctrl", "captions": "on"})

    def _cycle_voice(self):
        if not isinstance(self.tts, SileroTTS) or not self.voice_cycle:
            console.print("[yellow]Voice switch (F2) is available only with Silero TTS.[/yellow]")
            return
        self.voice_idx = (self.voice_idx + 1) % len(self.voice_cycle)
        new_voice = self.voice_cycle[self.voice_idx]
        self.tts.speaker = new_voice
        label = "male" if new_voice in self._male_voices else "female"
        console.print(f"[magenta]TTS voice → [bold]{new_voice}[/bold] ({label})[/magenta]")

    def _toggle_noise_gate(self):
        self.fast_gate.cfg.enable = not self.fast_gate.cfg.enable
        console.print(f"[magenta]Noise gate: {'ON' if self.fast_gate.cfg.enable else 'OFF'}[/magenta]")

    def _on_esc(self):
        now = time.time()
        if now - self._last_esc <= 1.2:
            console.print("[cyan]ESC ESC → exiting...[/cyan]")
            self.stop_event.set()
        else:
            self._last_esc = now
            console.print("[dim]Press ESC again to quit[/dim]")

    def _setup_hotkeys(self):
        if keyboard is None:
            console.print("[yellow]Hotkeys disabled (python 'keyboard' module not available). Use Ctrl+C to stop.[/yellow]")
            return
        keyboard.add_hotkey("f1", self._toggle_mute)
        keyboard.add_hotkey("f2", self._cycle_voice)
        keyboard.add_hotkey("f3", self._toggle_noise_gate)  # <-- NEW
        keyboard.add_hotkey("esc", self._on_esc)
        console.print("[dim]Hotkeys: F1 = mute/unmute • F2 = switch voice (male↔female) • F3 = noise gate on/off • ESC×2 = quit[/dim]")

    # ---------- Workers ----------
    def _mt_worker(self):
        while not self.stop_event.is_set():
            try:
                src_text, asr_rt, start, mid = self.translate_q.get(timeout=0.1)
            except queue.Empty:
                continue
            tgt_text = self.translator.translate(src_text)
            tgt_text = self.glossary.apply(tgt_text)
            tgt_text = self.faithnorm.apply(tgt_text)

            # Looser target gate; skip garbagey repeated outputs
            if self.fast_gate.drop_tgt(tgt_text):
                continue

            t1 = time.time()
            try:
                self.print_q.put_nowait((src_text, tgt_text, asr_rt, mid - start - asr_rt, t1 - mid, t1 - start))
            except queue.Full:
                pass

    def _tts_worker(self):
        """Runs in background; keeps audio flowing while ASR/MT continues."""
        while not self.stop_event.is_set():
            try:
                text = self.tts_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.tts_muted or not text.strip():
                continue
            try:
                audio, sr = self.tts.synthesize(text)
                if self.lan and audio.size:
                    self.lan.send(audio, sr)
            except Exception as e:
                console.print(f"[red]TTS error:[/red] {e}")

    def _enqueue_tts(self, text: str):
        if not text.strip() or self.tts_muted:
            return
        try:
            self.tts_q.put_nowait(text)
        except queue.Full:
            # drop if queue is full to keep latency low
            pass

    # ---------- Helpers ----------
    @staticmethod
    def _is_silent(x: np.ndarray, thresh_db: float = -45.0) -> bool:
        rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
        db = 20.0 * np.log10(rms)
        return db < thresh_db

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            console.log(f"[yellow]Audio status: {status}[/]")
        audio = np.clip(indata[:, 0], -1.0, 1.0)
        pcm16 = (audio * 32768.0).astype(np.int16).tobytes()
        seg = self.segmenter.process_block(pcm16)
        if seg is not None:
            self.q.put(seg)

    # ---------- Main run ----------
    def run(self):
        ac = self.audio_cfg
        sd.default.samplerate = ac.samplerate
        sd.default.channels = ac.channels
        sd.default.latency = 'low'
        if ac.device is not None:
            sd.default.device = ac.device

        self._setup_hotkeys()

        ip = guess_lan_ip()
        console.print(Panel.fit(
            "[bold green]Starting live translation[/bold green]\n"
            "Broadcast-only: your machine stays silent.\n"
            f"Open [b]http://{ip}:{self.lan_cfg.port}[/b] on devices in the same Wi-Fi and press Start.\n"
            "Press [b]Ctrl+C[/b] or [b]ESC twice[/b] to stop.\n"
            "[dim]F1: mute/unmute • F2: switch voice (male↔female) • F3: noise gate on/off • ESC×2: quit[/dim]",
            box=box.ROUNDED
        ))

        with sd.InputStream(callback=self.audio_callback, dtype="float32", blocksize=self.segmenter.block_size):
            try:
                while not self.stop_event.is_set():
                    # If muted, keep queues empty so resume is instant
                    if self.tts_muted:
                        self._drain_queue(self.q)
                        self._drain_queue(self.translate_q)
                        self._drain_queue(self.print_q)
                        self._drain_queue(self.tts_q)
                        time.sleep(0.01)
                        continue

                    # Drain completed translations for display / TTS
                    try:
                        while True:
                            src, tgt, asr_rt, wait_gap, mt_rt, total = self.print_q.get_nowait()
                            self._print_result(src, tgt, asr_rt, wait_gap, mt_rt, total)
                            if self.lan:
                                self.lan.send_text({
                                    "type":"trans",
                                    "src": src.strip(),
                                    "tgt": tgt.strip(),
                                    "asr_ms": int(round(asr_rt*1000)),
                                    "mt_ms": int(round(mt_rt*1000)),
                                    "total_ms": int(round(total*1000))
                                })
                            self._enqueue_tts(tgt)   # non-blocking, background TTS
                    except queue.Empty:
                        pass

                    # Get a new audio segment if available
                    try:
                        segment = self.q.get(timeout=0.02)
                    except queue.Empty:
                        continue

                    if LiveTranslatorApp._is_silent(segment, -45.0):
                        continue

                    start = time.time()
                    src_text, asr_rt, stats = self.transcriber.transcribe(segment, self.audio_cfg.samplerate)
                    mid = time.time()

                    # ASR guards
                    g = self.asr_guard
                    if g.enable:
                        if stats["avg_logprob"] < g.min_avg_logprob:    continue
                        if stats["compression_ratio"] > g.max_compression_ratio:  continue
                        if stats["no_speech_prob"] > g.max_no_speech_prob:        continue
                        if has_long_repeat_run(src_text, g.max_repeat_run):       continue
                        if g.dedupe_repeats: src_text = collapse_repeats(src_text)

                    # Basic guards
                    if not src_text or len(src_text.strip()) < 3 or len(src_text.split()) < 2:  continue
                    if not mostly_latin(src_text):  continue
                    core = src_text.strip().strip(".?!,:;—-")
                    if not core:  continue

                    # Deduplicate exact repeats (5s window)
                    if not hasattr(self, "_last_src"): self._last_src = ("", 0.0)
                    last_text, last_t = self._last_src
                    now = time.time()
                    norm = core.lower()
                    if norm == last_text and (now - last_t) < 5.0:  continue
                    self._last_src = (norm, now)

                    # Normalize scripture refs
                    src_text = fix_refs(src_text)

                    # FINAL fast gate on source — if bad, SKIP MT & TTS
                    if self.fast_gate.drop_src(src_text):
                        continue

                    # Send to MT worker (non-blocking)
                    try:
                        self.translate_q.put_nowait((src_text, asr_rt, start, mid))
                    except queue.Full:
                        pass

            except KeyboardInterrupt:
                console.print("\n[cyan]Stopping...[/cyan]")
                self.stop_event.set()
            finally:
                try:
                    if keyboard is not None:
                        keyboard.clear_all_hotkeys()
                except Exception:
                    pass
                try:
                    self.mt_thread.join(timeout=1.0)
                    self.tts_thread.join(timeout=1.0)
                except Exception:
                    pass

    def _print_result(self, src: str, tgt: str, asr_rt: float, wait_gap: float, mt_rt: float, total: float):
        if not src.strip():
            return
        tbl = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
        tbl.add_row("[bold]Source[/bold]", src.strip())
        tbl.add_row("[bold]Translation[/bold]", f"[green]{tgt.strip()}[/green]")
        timing = f"ASR {asr_rt*1000:.0f} ms • MT {mt_rt*1000:.0f} ms • Total {total*1000:.0f} ms"
        console.print(Panel(tbl, title=timing, border_style="blue"))

# --------------------- Utils ---------------------
def guess_lan_ip() -> str:
    ip = "127.0.0.1"; s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            pass
    finally:
        try:
            if s: s.close()
        except Exception:
            pass
    return ip

# --------------------- CLI ---------------------
def parse_args():
    p = argparse.ArgumentParser(description="Live offline EN→RU speech translation (broadcast-only over LAN)")
    p.add_argument("--tts-engine", choices=["silero", "piper"], default="silero", help="Choose TTS engine")
    p.add_argument("--tts-voice", default="aidar", help="TTS voice (silero: aidar/eugene/nikolay/baya/xenia)")
    p.add_argument("--female-voice", default="xenia", help="Female voice to toggle to (silero: baya/xenia)")
    p.add_argument("--asr-lang", default="en", help="Force ASR language ('en' or 'None' to autodetect)")
    p.add_argument("--whisper", default="small", help="Whisper size: tiny/base/small/medium/large-v3")
    p.add_argument("--device", default="auto", help="ASR device: auto|cuda|cpu")
    p.add_argument("--compute-type", choices=["float16", "int8", "int8_float16", "float32"], help="Whisper compute type override")
    p.add_argument("--beam", type=int, default=1, help="ASR beam size (1 = greedy fastest)")
    p.add_argument("--vad", action="store_true", help="Also enable Whisper's internal VAD")
    p.add_argument("--block-ms", type=int, choices=[10, 20, 30], default=20, help="Audio block size for VAD (10/20/30 ms)")
    p.add_argument("--silence-ms", type=int, default=320, help="Silence to end segment")
    p.add_argument("--max-segment-ms", type=int, default=8000, help="Hard cap for one utterance (prevents repetition loops)")
    p.add_argument("--vad-aggr", type=int, default=3, help="WebRTC VAD aggressiveness 0-3")
    p.add_argument("--glossary", type=str, default="glossary.ru.json", help="Glossary JSON path")
    p.add_argument("--mt-model", type=str, default="Helsinki-NLP/opus-mt-en-ru", help="MT model name (Marian or NLLB)")
    p.add_argument("--no-tts", action="store_true", help="Disable TTS (no broadcast)")
    p.add_argument("--piper-exe", type=str, default=os.getenv("PIPER_EXE"))
    p.add_argument("--piper-voice", type=str, default=os.getenv("VOICE_PATH"))
    p.add_argument("--lan-host", default="0.0.0.0", help="LAN bind host (default 0.0.0.0)")
    p.add_argument("--lan-port", type=int, default=8765, help="LAN HTTP/WebSocket port")
    p.add_argument("--no-lan", action="store_true", help="Disable LAN audio server")
    return p.parse_args()

def main():
    args = parse_args()
    a_cfg = AudioConfig(block_ms=args.block_ms, max_silence_ms=args.silence_ms,
                        vad_aggressiveness=args.vad_aggr, max_segment_ms=args.max_segment_ms)
    w_cfg = WhisperConfig(model_size=args.whisper, device=args.device, beam_size=args.beam, vad_filter=args.vad,
                          compute_type=args.compute_type,
                          language=None if (args.asr_lang is None or str(args.asr_lang).lower() == "none") else args.asr_lang)
    m_cfg = MTConfig(model_name=args.mt_model, src_lang=nllb_code("en"), tgt_lang=nllb_code("ru"))
    p_cfg = PiperConfig(enabled=not args.no_tts, exe_path=args.piper_exe, voice_path=args.piper_voice,
                        engine=args.tts_engine, tts_voice=args.tts_voice, female_voice=args.female_voice)
    lan_cfg = LANConfig(host=args.lan_host, port=args.lan_port, enable=not args.no_lan)
    app = LiveTranslatorApp(a_cfg, w_cfg, m_cfg, p_cfg, Path(args.glossary), lan_cfg)
    app.run()

if __name__ == "__main__":
    main()
