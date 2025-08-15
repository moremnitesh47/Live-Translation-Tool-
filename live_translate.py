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

@dataclass(slots=True)
class WhisperConfig:
    model_size: str = "small"    # tiny/base/small/medium/large-v3
    device: str = "auto"         # cuda|cpu|auto
    compute_type: Optional[str] = None  # float16|int8|int8_float16|float32
    beam_size: int = 1
    vad_filter: bool = False
    language: Optional[str] = "en"  # force English ASR by default
    initial_prompt: Optional[str] = (
        "Church sermon, biblical terms, Matthew, Mark, Luke, John, Paul, Holy Spirit, Lord's Prayer"
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
    # e.g., "Matt 5:7", "Matthew 5 7", "Rom. 10:9"
    (re.compile(fr"\b(?:{pat})\s*(?P<ch>\d+)\s*[:\s]\s*(?P<vs>\d+)\b", re.I), book)
    for pat, book in REF_BOOKS.items()
]

def fix_refs(s: str) -> str:
    out = s
    for pat, book in _ref_patterns:
        out = pat.sub(lambda m: f"{book} {m.group('ch')}:{m.group('vs')}", out)
    return out

# --------------------- Audio + VAD capture ---------------------
class VADSegmenter:
    def __init__(self, cfg: AudioConfig):
        if cfg.block_ms not in (10, 20, 30):
            raise ValueError("block_ms must be 10, 20, or 30 for WebRTC VAD")
        self.cfg = cfg
        self.vad = webrtcvad.Vad(cfg.vad_aggressiveness)
        self.block_size = int(cfg.samplerate * cfg.block_ms / 1000)
        self.silence_blocks = max(1, cfg.max_silence_ms // cfg.block_ms)
        self.frames: List[bytes] = []
        self.silent_count = 0

    def process_block(self, pcm16: bytes) -> Optional[np.ndarray]:
        is_speech = self.vad.is_speech(pcm16, sample_rate=self.cfg.samplerate)
        if is_speech:
            self.frames.append(pcm16)
            self.silent_count = 0
            return None
        else:
            if self.frames:
                self.silent_count += 1
                self.frames.append(pcm16)
                if self.silent_count >= self.silence_blocks:
                    chunk = b"".join(self.frames)
                    self.frames = []
                    self.silent_count = 0
                    audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    return audio
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

    def transcribe(self, audio: np.ndarray, samplerate: int) -> Tuple[str, float]:
        start = time.time()
        segments, info = self.model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad_filter,  # keep only once
            word_timestamps=False,
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.6,
            log_prob_threshold=-0.5,
            initial_prompt=self.cfg.initial_prompt,
        )
        text = "".join(seg.text for seg in segments).strip()
        return text, (time.time() - start)

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
    def __init__(self, speaker: str = "aidar", sample_rate: int = 48000):
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
                stdout, stderr = p.communicate(input=text.encode("utf-8"), timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate()
                raise RuntimeError("piper timed out")
            if p.returncode != 0:
                raise RuntimeError(f"piper failed (code {p.returncode}): {stderr.decode(errors='ignore')}")
            data, sr = sf.read(wav_path, dtype="float32")
            return data.astype(np.float32, copy=False), int(sr)
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

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

# --------------------- LAN audio+transcript server (WebSocket) ---------------------
class LANAudioServer:
    """
    Offline LAN broadcaster.
    - Serves / (HTML player) and /ws (WebSocket)
    - Broadcasts:
        * Audio chunks as binary: header b'PCM0' + uint32_le(sr) + uint32_le(n) + float32[n]
        * Transcript cards as text JSON: {"type":"trans", "src":..., "tgt":..., "asr_ms":..., "mt_ms":..., "total_ms":...}
    """
    HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LAN TTS</title>
  <style>
    :root {
      --bg: #fafafa;
      --fg: #222;
      --muted: #727272;
      --card: #fff;
      --accent: #007acc;
      --border: #dadada;
      --radius: 12px;
      --shadow-1: rgba(0,0,0,.06);
      --shadow-2: rgba(0,0,0,.12);
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0e0f12; --fg: #e9e9ea; --muted:#a0a0ad; --card:#14161a; --border:#282b31; --accent:#4aa3ff;
        --shadow-1: rgba(0,0,0,.4); --shadow-2: rgba(0,0,0,.6);
      }
    }

    *, *::before, *::after { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; }
    body { background: var(--bg); color: var(--fg); font-family: var(--font); font-size: 16px; line-height: 1.5; -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }

    header {
      position: sticky; top: 0; z-index: 10;
      background: var(--card);
      border-bottom: 1px solid var(--border);
      box-shadow: 0 2px 6px var(--shadow-1);
      padding: 14px 20px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between;
    }
    .row { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }

    button { cursor: pointer; border: 0; border-radius: var(--radius); background: var(--accent); color: #fff; padding: 12px 18px; font-weight: 600; font-size: 1rem; box-shadow: 0 2px 4px var(--shadow-2); transition: transform .1s ease, box-shadow .2s ease, background-color .25s ease; user-select: none; -webkit-tap-highlight-color: transparent; }
    button:hover:not(:disabled), button:focus-visible:not(:disabled) { background: #005ea3; box-shadow: 0 4px 12px var(--shadow-2); transform: translateY(-1px); }
    button:disabled { opacity:.65; cursor: default; box-shadow: none; }
    button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

    label { font-size: .95rem; color: var(--muted); display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
    input[type="checkbox"] { width: 18px; height: 18px; }

    main { max-width: 900px; margin: 20px auto 40px; padding: 0 16px; width: 100%; }

    #status { font-size: .9rem; color: var(--muted); min-width: 120px; white-space: nowrap; }

    #feed { margin-top: 20px; display: grid; gap: 14px; width: 100%; }

    .card { background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 16px; box-shadow: 0 1px 4px var(--shadow-1); word-break: break-word; overflow-wrap: anywhere; font-size: 1.05rem; transition: box-shadow .2s ease; }
    .card:hover { box-shadow: 0 6px 16px var(--shadow-2); }

    .meta { font-size: .85rem; color: var(--muted); margin-bottom: 8px; }
    .kv .k { color: #444; font-weight: 600; }
    .kv .v { color: var(--accent); margin-bottom: 4px; font-weight: 500; }

    .hidden { display: none !important; }

    pre.log { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 10px; max-height: 220px; overflow: auto; font-size: .95rem; margin-top: 18px; }

    @media (max-width: 600px) {
      header, .row { flex-direction: column; align-items: stretch; }
      button, label { width: 100%; font-size: 1.05rem; padding: 14px 0; justify-content: center; }
      #status { text-align: center; min-width: auto; white-space: normal; }
      .card { font-size: 1.1rem; }
    }
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

  <script>
    // ===== Helpers & State =====
    const $ = sel => document.querySelector(sel);
    const FEED_MAX = 50, MAGIC = 0x50434d30; // 'PCM0'

    let ctx = null, node = null, ws = null, wakeLock = null;
    let cur = null, idx = 0, playing = false, joined = false;
    let logEnabled = false, transEnabled = true;
    let reconnectTimer = null, backoff = 1000;

    const setStatus = t => { const s = $('#status'); if (s) s.textContent = t; };
    const log = m => { if (!logEnabled) return; const el = $('#log'); el.textContent += m + "\n"; el.scrollTop = el.scrollHeight; };

    // ===== Wake Lock =====
    async function requestWakeLock(){ if(!('wakeLock' in navigator)) { log('Wake Lock API not available'); return; }
      try{ wakeLock = await navigator.wakeLock.request('screen'); wakeLock.addEventListener('release', ()=>log('Wake lock released')); log('Wake lock active'); }catch(e){ log('Wake lock failed: '+(e?.message||e)); }
    }
    async function releaseWakeLock(){ try{ if(wakeLock){ await wakeLock.release(); wakeLock = null; } }catch{ /* noop */ } }

    // ===== Audio =====
    function setupAudio(){ if(ctx) return;
      try { ctx = new (window.AudioContext||window.webkitAudioContext)({ sampleRate: 48000 }); }
      catch { ctx = new (window.AudioContext||window.webkitAudioContext)(); }
      node = (ctx.createScriptProcessor||ctx.createJavaScriptNode)?.call(ctx, 2048, 0, 1);
      node.onaudioprocess = e => {
        const out = e.outputBuffer.getChannelData(0); let i = 0, n = out.length;
        while(i < n){
          if(!cur || idx >= cur.length){ cur = (queue.length ? queue.shift() : null); idx = 0; if(!cur){ out.fill(0, i); if(playing){ playing=false; setStatus('Buffering…'); } break; } }
          const take = Math.min(n - i, cur.length - idx);
          out.set(cur.subarray(idx, idx + take), i);
          i += take; idx += take; if(!playing){ playing = true; setStatus(`Playing @ ${ctx.sampleRate} Hz…`); }
        }
      };
      node.connect(ctx.destination);
    }

    function teardownAudio(){ try{ node?.disconnect(); }catch{}; try{ ctx?.close(); }catch{}; node=null; ctx=null; cur=null; idx=0; playing=false; queue.length=0; }

    function resampleFloat32(data, srcRate, dstRate){ if(srcRate===dstRate) return data; const ratio = srcRate/dstRate; const newLen = Math.max(1, Math.round(data.length/ratio)); const out = new Float32Array(newLen); for(let i=0;i<newLen;i++){ const j=i*ratio, j0=j|0, frac=j-j0; const a=data[j0]||0, b=data[j0+1]??a; out[i]=a+frac*(b-a); } return out; }

    // ===== UI: transcript cards =====
    function addCard(msg){ if(!transEnabled) return; const feed = $('#feed'); const card = document.createElement('div'); card.className='card';
      card.innerHTML = `<div class="meta">ASR ${msg.asr_ms} ms • MT ${msg.mt_ms} ms • Total ${msg.total_ms} ms</div>
                        <div class="kv"><div class="k">Source</div><div class="v">${msg.src||''}</div>
                        <div class="k">Translation</div><div class="v">${msg.tgt||''}</div></div>`;
      feed.append(card);
      while(feed.childElementCount > FEED_MAX) feed.firstElementChild.remove();
    }

    // ===== WebSocket =====
    const queue = [];
    function startWS(){ if(ws) return;
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => { setStatus('Connected. Waiting for audio…'); log('WebSocket connected'); backoff = 1000; };
      ws.onclose = () => { log('WebSocket disconnected'); ws = null; setStatus('Disconnected.'); if(joined) scheduleReconnect(); };
      ws.onerror = e => { log('WebSocket error'); };
      ws.onmessage = ev => {
        try{
          if(typeof ev.data === 'string') { const msg = JSON.parse(ev.data); if(msg?.type === 'trans') addCard(msg); return; }
          const dv = new DataView(ev.data);
          if(dv.getUint32(0, false) !== MAGIC){ log('Unknown binary payload'); return; }
          const sr = dv.getUint32(4, true); // little-endian
          const f32 = new Float32Array(ev.data, 12);
          const data = ctx ? resampleFloat32(f32, sr, ctx.sampleRate) : f32;
          queue.push(data);
          if(!playing && ctx) setStatus(`Playing @ ${ctx.sampleRate} Hz…`);
        }catch(ex){ log('WS message error: '+ex.message); }
      };
    }

    function scheduleReconnect(){ if(reconnectTimer) return; reconnectTimer = setTimeout(()=>{ reconnectTimer=null; if(!ws && joined){ log('Reconnecting…'); startWS(); backoff = Math.min(backoff*1.6, 8000); } }, backoff); }

    function stopAll(){ joined=false; try{ ws?.close(); }catch{} ws=null; clearTimeout(reconnectTimer); reconnectTimer=null; releaseWakeLock(); teardownAudio(); setStatus('Idle'); $('#btn').textContent = 'Join the Russian Channel'; }

    // ===== Events =====
    $('#btn').onclick = async () => {
      if(!joined){ joined = true; $('#btn').textContent = 'Leave channel'; setupAudio(); startWS(); if($('#awakeToggle').checked) await requestWakeLock(); }
      else { stopAll(); }
    };

    document.addEventListener('visibilitychange', async () => { if(document.visibilityState === 'visible' && $('#awakeToggle').checked) await requestWakeLock(); });

    $('#logToggle').onchange = e => { logEnabled = e.target.checked; $('#log').classList.toggle('hidden', !logEnabled); if(logEnabled) log('Debug log enabled'); };
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
        """Queue one audio chunk (float32 mono). Non-blocking."""
        try:
            self._q.put_nowait(('A', samples.astype(np.float32, copy=False), int(sample_rate)))
        except queue.Full:
            pass

    def send_text(self, payload: dict):
        """Queue one transcript card as JSON string."""
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

        # On Windows, ensure a selector loop in this background thread
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
                        try:
                            await ws.send_bytes(payload)
                        except Exception:
                            dead.append(ws)
                elif kind == 'T':
                    _, js = item
                    for ws in list(websockets):
                        try:
                            await ws.send_str(js)
                        except Exception:
                            dead.append(ws)
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

        # --- TTS selection (Silero default) ---
        if p_cfg.engine == "silero" and p_cfg.enabled is not False:
            self.tts = SileroTTS(speaker=p_cfg.tts_voice)
        elif p_cfg.engine == "piper" and p_cfg.enabled is not False:
            self.tts = PiperTTS(p_cfg)
        else:
            class _NoTTS:
                def synthesize(self, text: str): return (np.zeros(0, dtype=np.float32), 24000)
            self.tts = _NoTTS()

        # --- Hotkey-related state ---
        self.tts_muted = False
        self._last_esc = 0.0

        # Voice toggle: current male ↔ chosen female (Silero only)
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
            self.voice_cycle = []
            self.voice_idx = 0

        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.translate_q: queue.Queue[Tuple[str, float, float, float]] = queue.Queue(maxsize=4)
        self.print_q: queue.Queue[Tuple[str, str, float, float, float, float]] = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        # Warm-up
        _ = self.translator.translate("Hello")
        dummy = np.zeros(int(self.audio_cfg.samplerate * 0.5), dtype=np.float32)
        _ = self.transcriber.transcribe(dummy, self.audio_cfg.samplerate)
        # MT worker
        self.mt_thread = threading.Thread(target=self._mt_worker, daemon=True)
        self.mt_thread.start()

        # --- LAN server ---
        self.lan = LANAudioServer(lan_cfg) if lan_cfg.enable else None
        self.lan_cfg = lan_cfg

    # ---------- Hotkeys ----------
    def _toggle_mute(self):
        self.tts_muted = not self.tts_muted
        state = "muted" if self.tts_muted else "unmuted"
        console.print(f"[magenta]Broadcast: [bold]{state}[/bold].[/magenta]")

    def _cycle_voice(self):
        if not isinstance(self.tts, SileroTTS) or not self.voice_cycle:
            console.print("[yellow]Voice switch (F2) is available only with Silero TTS.[/yellow]")
            return
        self.voice_idx = (self.voice_idx + 1) % len(self.voice_cycle)
        new_voice = self.voice_cycle[self.voice_idx]
        self.tts.speaker = new_voice
        label = "male" if new_voice in self._male_voices else "female"
        console.print(f"[magenta]TTS voice → [bold]{new_voice}[/bold] ({label})[/magenta]")

    def _on_esc(self):
        # Double-press ESC (within 1.2s) to quit
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
        keyboard.add_hotkey("esc", self._on_esc)
        console.print("[dim]Hotkeys: F1 = mute/unmute • F2 = switch voice (male↔female) • ESC×2 = quit[/dim]")

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

    def _mt_worker(self):
        while not self.stop_event.is_set():
            try:
                src_text, asr_rt, start, mid = self.translate_q.get(timeout=0.1)
            except queue.Empty:
                continue
            tgt_text = self.translator.translate(src_text)
            tgt_text = self.glossary.apply(tgt_text)
            t1 = time.time()
            self.print_q.put((src_text, tgt_text, asr_rt, mid - start - asr_rt, t1 - mid, t1 - start))

    def _speak(self, text: str):
        if self.tts_muted or not text.strip():
            return
        try:
            audio, sr = self.tts.synthesize(text)
            if self.lan and audio.size:
                self.lan.send(audio, sr)
        except Exception as e:
            console.print(f"[red]TTS error:[/red] {e}")

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
            "Open [b]http://%s:%d[/b] on devices in the same Wi-Fi and press Start.\n"
            "Press [b]Ctrl+C[/b] or [b]ESC twice[/b] to stop.\n"
            "[dim]F1: mute/unmute • F2: switch voice (male↔female) • ESC×2: quit[/dim]" % (ip, self.lan_cfg.port),
            box=box.ROUNDED
        ))

        with sd.InputStream(callback=self.audio_callback, dtype="float32", blocksize=self.segmenter.block_size):
            try:
                while not self.stop_event.is_set():
                    # Drain completed translations for display / TTS
                    try:
                        while True:
                            src, tgt, asr_rt, wait_gap, mt_rt, total = self.print_q.get_nowait()
                            self._print_result(src, tgt, asr_rt, wait_gap, mt_rt, total)
                            # Broadcast transcript to web clients (tiny JSON)
                            if self.lan:
                                self.lan.send_text({
                                    "type":"trans",
                                    "src": src.strip(),
                                    "tgt": tgt.strip(),
                                    "asr_ms": int(round(asr_rt*1000)),
                                    "mt_ms": int(round(mt_rt*1000)),
                                    "total_ms": int(round(total*1000))
                                })
                            self._speak(tgt)
                    except queue.Empty:
                        pass

                    # Get a new audio segment if available
                    try:
                        segment = self.q.get(timeout=0.05)
                    except queue.Empty:
                        continue

                    if LiveTranslatorApp._is_silent(segment, -45.0):
                        continue

                    start = time.time()
                    src_text, asr_rt = self.transcriber.transcribe(segment, self.audio_cfg.samplerate)
                    mid = time.time()

                    # Guards: skip junk, duplicates, non-English
                    if not src_text or len(src_text.strip()) < 3 or len(src_text.split()) < 2:
                        continue
                    if not mostly_latin(src_text):
                        continue
                    core = src_text.strip().strip(".?!,:;—-")
                    if not core:
                        continue
                    # Deduplicate exact repeats (5s window)
                    if not hasattr(self, "_last_src"):
                        self._last_src = ("", 0.0)
                    last_text, last_t = self._last_src
                    now = time.time()
                    norm = core.lower()
                    if norm == last_text and (now - last_t) < 5.0:
                        continue
                    self._last_src = (norm, now)

                    # Normalize scripture refs
                    src_text = fix_refs(src_text)

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
    """Best-effort LAN IP for user hinting."""
    ip = "127.0.0.1"
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # doesn't need to be reachable
        ip = s.getsockname()[0]
    except Exception:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            pass
    finally:
        try:
            if s:
                s.close()
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
    a_cfg = AudioConfig(block_ms=args.block_ms, max_silence_ms=args.silence_ms, vad_aggressiveness=args.vad_aggr)
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





# from __future__ import annotations
# import os, sys, re, time, json, queue, argparse, threading, socket, glob
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Optional, List, Tuple

# # ---------- Environment speedups (set BEFORE heavy imports) ----------
# os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))
# os.environ.setdefault("MKL_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# # Windows: add CUDA bin dirs if present so DLLs load when you switch to --device cuda
# if sys.platform.startswith("win"):
#     for bin_dir in glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin"):
#         try:
#             os.add_dll_directory(bin_dir)
#         except Exception:
#             pass

# import numpy as np
# import sounddevice as sd
# import webrtcvad
# from rich.console import Console
# from rich.table import Table
# from rich.panel import Panel
# from rich import box
# from faster_whisper import WhisperModel
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# # --- Optional: global hotkeys (F1/F2) ---
# try:
#     import keyboard  # pip install keyboard
# except Exception:
#     keyboard = None  # hotkeys will be disabled if unavailable

# console = Console()

# # --------------------- Language utilities ---------------------
# NLLB_LANG = {"en": "eng_Latn", "ru": "rus_Cyrl", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn"}

# def nllb_code(lang: str) -> str:
#     return NLLB_LANG.get(lang.lower(), "eng_Latn")

# # --------------------- Config ---------------------
# @dataclass(slots=True)
# class AudioConfig:
#     samplerate: int = 16000
#     channels: int = 1
#     block_ms: int = 20  # must be 10, 20, or 30 for WebRTC VAD
#     vad_aggressiveness: int = 3
#     max_silence_ms: int = 320
#     device: Optional[int] = None

# @dataclass(slots=True)
# class WhisperConfig:
#     model_size: str = "small"    # tiny/base/small/medium/large-v3
#     device: str = "auto"         # cuda|cpu|auto
#     compute_type: Optional[str] = None  # float16|int8|int8_float16|float32
#     beam_size: int = 1
#     vad_filter: bool = False
#     language: Optional[str] = "en"  # force English ASR by default
#     initial_prompt: Optional[str] = (
#         "Church sermon, biblical terms, Matthew, Mark, Luke, John, Paul, Holy Spirit, Lord's Prayer"
#     )

# @dataclass(slots=True)
# class MTConfig:
#     model_name: str = "Helsinki-NLP/opus-mt-en-ru"  # fast, accurate on CPU
#     src_lang: str = "eng_Latn"
#     tgt_lang: str = "rus_Cyrl"
#     max_new_tokens: int = 128

# @dataclass(slots=True)
# class PiperConfig:
#     enabled: bool = False
#     exe_path: Optional[str] = None
#     voice_path: Optional[str] = None
#     engine: str = "silero"            # "silero" or "piper"
#     tts_voice: str = "aidar"          # silero voices: aidar/eugene/nikolay/baya/xenia

# # --------------------- LAN Audio Server Config ---------------------
# @dataclass(slots=True)
# class LANConfig:
#     host: str = "0.0.0.0"   # listen on all interfaces
#     port: int = 8765        # http://<server-ip>:8765
#     enable: bool = True

# # --------------------- Helpers ---------------------
# _LATIN_RATIO_MIN = 0.85

# def mostly_latin(s: str) -> bool:
#     if not s:
#         return False
#     letters = [ch for ch in s if ch.isalpha()]
#     if not letters:
#         return False
#     latin = sum('A' <= ch <= 'Z' or 'a' <= ch <= 'z' for ch in letters)
#     return (latin / max(1, len(letters))) >= _LATIN_RATIO_MIN

# # Robust scripture reference patterns
# REF_BOOKS = {
#     r"Matt(?:hew)?\.?": "Matthew",
#     r"Mark\.?": "Mark",
#     r"Luke\.?": "Luke",
#     r"John\.?": "John",
#     r"Rom(?:ans)?\.?": "Romans",
# }
# _ref_patterns = [
#     # e.g., "Matt 5:7", "Matthew 5 7", "Rom. 10:9"
#     (re.compile(fr"\b(?:{pat})\s*(?P<ch>\d+)\s*[:\s]\s*(?P<vs>\d+)\b", re.I), book)
#     for pat, book in REF_BOOKS.items()
# ]

# def fix_refs(s: str) -> str:
#     out = s
#     for pat, book in _ref_patterns:
#         out = pat.sub(lambda m: f"{book} {m.group('ch')}:{m.group('vs')}", out)
#     return out

# # --------------------- Audio + VAD capture ---------------------
# class VADSegmenter:
#     def __init__(self, cfg: AudioConfig):
#         if cfg.block_ms not in (10, 20, 30):
#             raise ValueError("block_ms must be 10, 20, or 30 for WebRTC VAD")
#         self.cfg = cfg
#         self.vad = webrtcvad.Vad(cfg.vad_aggressiveness)
#         self.block_size = int(cfg.samplerate * cfg.block_ms / 1000)
#         # ensure at least one block of silence
#         self.silence_blocks = max(1, cfg.max_silence_ms // cfg.block_ms)
#         self.frames: List[bytes] = []
#         self.silent_count = 0

#     def process_block(self, pcm16: bytes) -> Optional[np.ndarray]:
#         is_speech = self.vad.is_speech(pcm16, sample_rate=self.cfg.samplerate)
#         if is_speech:
#             self.frames.append(pcm16)
#             self.silent_count = 0
#             return None
#         else:
#             if self.frames:
#                 self.silent_count += 1
#                 self.frames.append(pcm16)
#                 if self.silent_count >= self.silence_blocks:
#                     chunk = b"".join(self.frames)
#                     self.frames = []
#                     self.silent_count = 0
#                     audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
#                     return audio
#             return None

# # --------------------- ASR (faster-whisper) ---------------------
# class Transcriber:
#     def __init__(self, cfg: WhisperConfig):
#         device = cfg.device
#         if device == "auto":
#             device = "cuda" if _has_cuda() else "cpu"
#         compute_type = cfg.compute_type or ("float16" if device == "cuda" else "int8")
#         console.print(f"[cyan]Loading Whisper '{cfg.model_size}' on {device} ({compute_type})...[/cyan]")
#         self.model = WhisperModel(
#             cfg.model_size,
#             device=device,
#             compute_type=compute_type,
#             cpu_threads=max(1, (os.cpu_count() or 4) - 1)
#         )
#         self.cfg = cfg

#     def transcribe(self, audio: np.ndarray, samplerate: int) -> Tuple[str, float]:
#         start = time.time()
#         segments, info = self.model.transcribe(
#             audio,
#             language=self.cfg.language,
#             beam_size=self.cfg.beam_size,
#             vad_filter=self.cfg.vad_filter,  # keep only once
#             word_timestamps=False,
#             condition_on_previous_text=False,
#             temperature=0.0,
#             no_speech_threshold=0.6,
#             log_prob_threshold=-0.5,
#             initial_prompt=self.cfg.initial_prompt,
#         )
#         text = "".join(seg.text for seg in segments).strip()
#         return text, (time.time() - start)

# def _has_cuda() -> bool:
#     try:
#         import ctranslate2  # used by faster-whisper
#         return getattr(ctranslate2, "get_cuda_device_count", lambda: 0)() > 0
#     except Exception:
#         # fallback: respect CUDA_VISIBLE_DEVICES if explicitly set
#         return bool(os.environ.get("CUDA_VISIBLE_DEVICES"))

# # --------------------- Translator (Marian or NLLB) ---------------------
# class Translator:
#     def __init__(self, cfg: MTConfig):
#         self.cfg = cfg
#         console.print(f"[cyan]Loading translator '{cfg.model_name}'...[/cyan]")
#         self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name, use_safetensors=True)

#         if "opus-mt-" in cfg.model_name:
#             task = "translation_en_to_ru"
#             self.pipe = pipeline(
#                 task,
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 max_new_tokens=min(128, self.cfg.max_new_tokens),
#                 truncation=True,
#                 num_beams=1,
#                 no_repeat_ngram_size=3,
#                 length_penalty=1.0,
#             )
#         else:
#             self.pipe = pipeline(
#                 "translation",
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 src_lang=cfg.src_lang,
#                 tgt_lang=cfg.tgt_lang,
#                 max_new_tokens=self.cfg.max_new_tokens,
#                 truncation=True,
#             )

#     def translate(self, text: str) -> str:
#         if not text:
#             return ""
#         return self.pipe(text)[0]["translation_text"]

# # --------------------- Silero TTS (CPU, offline) ---------------------
# class SileroTTS:
#     """
#     CPU-friendly Russian TTS using Silero (offline).
#     Voices: 'aidar' (male), 'eugene' (male), 'nikolay' (male), 'baya' (female), 'xenia' (female)
#     """
#     def __init__(self, speaker: str = "aidar", sample_rate: int = 48000):
#         import torch
#         self.torch = torch
#         self.sample_rate = int(sample_rate)
#         self.speaker = speaker
#         # Load the small RU multi-speaker Silero model (CPU)
#         self.model, _ = torch.hub.load(
#             'snakers4/silero-models', 'silero_tts',
#             language='ru', speaker='v3_1_ru', trust_repo=True
#         )

#     def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
#         if not text or not text.strip():
#             return np.zeros(0, dtype=np.float32), self.sample_rate
#         with self.torch.no_grad():
#             audio = self.model.apply_tts(text=text, speaker=self.speaker, sample_rate=self.sample_rate)
#         return np.array(audio, dtype=np.float32), self.sample_rate

# # --------------------- Optional Piper TTS ---------------------
# class PiperTTS:
#     def __init__(self, cfg: PiperConfig):
#         self.enabled = bool(cfg.enabled and cfg.exe_path and cfg.voice_path
#                             and Path(cfg.exe_path).exists() and Path(cfg.voice_path).exists())
#         self.exe = cfg.exe_path
#         self.voice = cfg.voice_path

#     def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
#         if not (self.enabled and text and text.strip()):
#             return np.zeros(0, dtype=np.float32), 22050
#         import subprocess, tempfile, soundfile as sf, os
#         wav_path = None
#         try:
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
#                 wav_path = wf.name
#             cmd = [self.exe, "-m", self.voice, "-f", wav_path, "-q"]
#             p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             try:
#                 stdout, stderr = p.communicate(input=text.encode("utf-8"), timeout=30)
#             except subprocess.TimeoutExpired:
#                 p.kill()
#                 p.communicate()
#                 raise RuntimeError("piper timed out")
#             if p.returncode != 0:
#                 raise RuntimeError(f"piper failed (code {p.returncode}): {stderr.decode(errors='ignore')}")
#             data, sr = sf.read(wav_path, dtype="float32")
#             return data.astype(np.float32, copy=False), int(sr)
#         finally:
#             if wav_path and os.path.exists(wav_path):
#                 try:
#                     os.remove(wav_path)
#                 except Exception:
#                     pass

# # --------------------- Glossary ---------------------
# class Glossary:
#     def __init__(self, path: Path, tgt_lang: str):
#         self.map = {}
#         if path.exists():
#             try:
#                 data = json.loads(path.read_text(encoding="utf-8"))
#                 if data.get("target_lang") in (tgt_lang, None):
#                     self.map = data.get("replacements", {})
#             except Exception as e:
#                 console.print(f"[yellow]Failed to load glossary ({e}). Continuing without it.[/yellow]")

#     def apply(self, text: str) -> str:
#         if not self.map or not text:
#             return text
#         out = text
#         # Replace longer keys first; use word boundaries
#         for k in sorted(self.map, key=len, reverse=True):
#             v = self.map[k]
#             out = re.sub(rf"\b{re.escape(k)}\b", v, out)
#         return out

# # --------------------- LAN audio server (WebSocket) ---------------------
# class LANAudioServer:
#     """
#     Simple offline LAN broadcaster.
#     - Serves / (HTML player) and /ws (WebSocket)
#     - Broadcasts float32 PCM frames with a tiny binary header:
#         b'PCM0' + uint32_le(sample_rate) + uint32_le(n_samples) + float32[n_samples]
#     """
#     HTML_PAGE = r"""<!doctype html>
# <html lang="en"><meta charset="utf-8"><title>LAN TTS</title>
# <style>body{font-family:system-ui;margin:2rem}button{padding:.6rem 1rem}#log{margin-top:2rem;white-space:pre-wrap;background:#f0f0f0;padding:1rem;max-height:300px;overflow-y:auto;border:1px solid #ccc;border-radius:4px;font-size:0.9rem;}</style>
# <h2>LAN TTS Audio</h2>
# <p>Press Start (a user gesture is required for audio). This works offline on the same Wi-Fi/LAN.</p>
# <button id="btn">Start</button>
# <p id="status">Idle</p>
# <div><label><input type="checkbox" id="logToggle"> Enable debug log</label></div>
# <div id="log" style="display:none;"></div>
# <script>
# let ctx, node, queue=[], cur=null, idx=0, playing=false, ws=null, logEnabled=false;

# function log(msg){
#   if(logEnabled){
#     let logArea=document.getElementById('log');
#     logArea.textContent+=msg+"\n";
#     logArea.scrollTop=logArea.scrollHeight;
#   }
# }

# function setupAudio(){
#   if(ctx) return;
#   try{
#     ctx = new (window.AudioContext||window.webkitAudioContext)({ sampleRate: 48000 });
#     log('AudioContext created with sampleRate: '+ctx.sampleRate);
#   }catch(e){
#     ctx = new (window.AudioContext||window.webkitAudioContext)();
#     log('AudioContext created with default sampleRate');
#   }
#   node = ctx.createScriptProcessor(2048, 0, 1);
#   node.onaudioprocess = e=>{
#     let out = e.outputBuffer.getChannelData(0);
#     for(let i=0;i<out.length;i++){
#       if(!cur || idx>=cur.length){ cur = queue.length ? queue.shift() : null; idx = 0; }
#       out[i] = (cur ? cur[idx++] : 0);
#     }
#   };
#   node.connect(ctx.destination);
# }

# function resampleFloat32(data, srcRate, dstRate){
#   if(srcRate===dstRate) return data;
#   const ratio = srcRate/dstRate;
#   const newLen = Math.max(1, Math.round(data.length/ratio));
#   const out=new Float32Array(newLen);
#   for(let i=0;i<newLen;i++){
#     const idx = i*ratio;
#     const i0 = Math.floor(idx);
#     const frac = idx - i0;
#     const s0 = data[i0]||0;
#     const s1 = data[i0+1]||s0;
#     out[i] = s0+frac*(s1-s0);
#   }
#   return out;
# }

# function startWS(){
#   if(ws) return;
#   const proto = location.protocol === 'https:' ? 'wss' : 'ws';
#   ws = new WebSocket(proto + '://' + location.host + '/ws');
#   ws.binaryType = 'arraybuffer';
#   ws.onopen = ()=> { document.getElementById('status').innerText='Connected. Waiting for audio…'; log('WebSocket connected'); };
#   ws.onclose = ()=> { document.getElementById('status').innerText='Disconnected.'; log('WebSocket disconnected'); };
#   ws.onerror = e=> { document.getElementById('status').innerText='Error.'; log('WebSocket error: '+e.message); };
#   ws.onmessage = ev=>{
#     try {
#       const dv = new DataView(ev.data);
#       if(dv.getUint8(0)!==0x50 || dv.getUint8(1)!==0x43 || dv.getUint8(2)!==0x4D || dv.getUint8(3)!==0x30){
#         log('Invalid header received');
#         return;
#       }
#       const sr = dv.getUint32(4,true);
#       const f32 = new Float32Array(ev.data, 12);
#       const data = resampleFloat32(f32, sr, ctx.sampleRate);
#       queue.push(data);
#       if(!playing){ playing=true; document.getElementById('status').innerText=`Playing @ ${ctx.sampleRate} Hz…`; }
#       log('Received audio chunk with sampleRate: ' + sr + ', length: ' + f32.length);
#     } catch(ex) {
#       log('Error processing websocket message: ' + ex.message);
#     }
#   };
# }

# document.getElementById('btn').onclick = ()=>{
#   setupAudio();
#   startWS();
#   document.getElementById('btn').disabled=true;
# };

# document.getElementById('logToggle').onchange = e => {
#   logEnabled = e.target.checked;
#   document.getElementById('log').style.display = logEnabled ? 'block' : 'none';
#   if(logEnabled) log('Debug log enabled');
# }
# </script>
# </html>

# """

#     def __init__(self, cfg: LANConfig):
#         self.cfg = cfg
#         self._q: "queue.Queue[Tuple[np.ndarray,int]]" = queue.Queue(maxsize=32)
#         self._thread = threading.Thread(target=self._run, daemon=True)
#         self._thread.start()

#     def send(self, samples: np.ndarray, sample_rate: int):
#         """Queue one chunk (float32 mono). Non-blocking."""
#         try:
#             self._q.put_nowait((samples.astype(np.float32, copy=False), int(sample_rate)))
#         except queue.Full:
#             pass

#     def _run(self):
#         import asyncio, struct
#         from contextlib import suppress
#         try:
#             import aiohttp
#             from aiohttp import web
#         except Exception as e:
#             console.print(f"[red]aiohttp not installed: {e}. LAN broadcast disabled.[/red]")
#             return

#         # On Windows, ensure a selector loop in this background thread
#         try:
#             if sys.platform.startswith("win"):
#                 asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#         except Exception:
#             pass

#         async def index(_):
#             return web.Response(text=self.HTML_PAGE, content_type="text/html")

#         websockets: "set[web.WebSocketResponse]" = set()

#         async def ws_handler(request):
#             ws = web.WebSocketResponse()
#             await ws.prepare(request)
#             websockets.add(ws)
#             try:
#                 async for _ in ws:
#                     pass
#             finally:
#                 websockets.discard(ws)
#             return ws

#         async def broadcaster():
#             loop = asyncio.get_running_loop()
#             while True:
#                 samples, sr = await loop.run_in_executor(None, self._q.get)
#                 header = struct.pack("<4sII", b"PCM0", sr, samples.shape[0])
#                 payload = header + samples.tobytes(order="C")
#                 dead = []
#                 for ws in list(websockets):
#                     try:
#                         await ws.send_bytes(payload)
#                     except Exception:
#                         dead.append(ws)
#                 for d in dead:
#                     websockets.discard(d)

#         async def on_startup(app):
#             app['broadcaster_task'] = asyncio.create_task(broadcaster())
#             console.print(f"[green]LAN server listening on http://{guess_lan_ip()}:{self.cfg.port}[/green]")

#         async def on_cleanup(app):
#             task = app.get('broadcaster_task')
#             if task:
#                 task.cancel()
#                 with suppress(asyncio.CancelledError):
#                     await task

#         app = web.Application()
#         app.router.add_get("/", index)
#         app.router.add_get("/ws", ws_handler)
#         app.on_startup.append(on_startup)
#         app.on_cleanup.append(on_cleanup)

#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         runner = web.AppRunner(app)
#         loop.run_until_complete(runner.setup())
#         site = web.TCPSite(runner, self.cfg.host, self.cfg.port)
#         try:
#             loop.run_until_complete(site.start())
#         except OSError as e:
#             console.print(f"[red]Failed to bind {self.cfg.host}:{self.cfg.port} — {e}[/red]")
#             return
#         try:
#             loop.run_forever()
#         finally:
#             loop.run_until_complete(runner.cleanup())
#             loop.close()

# # --------------------- Main App with concurrency ---------------------
# class LiveTranslatorApp:
#     def __init__(self, a_cfg: AudioConfig, w_cfg: WhisperConfig, m_cfg: MTConfig, p_cfg: PiperConfig, glossary_path: Path, lan_cfg: LANConfig):
#         self.audio_cfg = a_cfg
#         self.segmenter = VADSegmenter(a_cfg)
#         self.transcriber = Transcriber(w_cfg)
#         self.translator = Translator(m_cfg)
#         self.glossary = Glossary(glossary_path, m_cfg.tgt_lang)

#         # --- TTS selection (Silero default) ---
#         if p_cfg.engine == "silero" and p_cfg.enabled is not False:
#             self.tts = SileroTTS(speaker=p_cfg.tts_voice)
#         elif p_cfg.engine == "piper" and p_cfg.enabled is not False:
#             self.tts = PiperTTS(p_cfg)
#         else:
#             class _NoTTS:
#                 def synthesize(self, text: str): return (np.zeros(0, dtype=np.float32), 24000)
#             self.tts = _NoTTS()

#         # --- Hotkey-related state ---
#         self.tts_muted = False  # controls broadcast now
#         self.voice_cycle = ["aidar", "eugene", "nikolay", "baya", "xenia"]
#         if isinstance(self.tts, SileroTTS) and getattr(self.tts, "speaker", None) in self.voice_cycle:
#             self.voice_idx = self.voice_cycle.index(self.tts.speaker)
#         else:
#             self.voice_idx = 0

#         self.q: queue.Queue[np.ndarray] = queue.Queue()
#         self.translate_q: queue.Queue[Tuple[str, float, float, float]] = queue.Queue(maxsize=4)
#         self.print_q: queue.Queue[Tuple[str, str, float, float, float, float]] = queue.Queue(maxsize=8)
#         self.stop_event = threading.Event()
#         # Warm-up (removes 1st-use spikes)
#         _ = self.translator.translate("Hello")
#         dummy = np.zeros(int(self.audio_cfg.samplerate * 0.5), dtype=np.float32)
#         _ = self.transcriber.transcribe(dummy, self.audio_cfg.samplerate)
#         # MT worker
#         self.mt_thread = threading.Thread(target=self._mt_worker, daemon=True)
#         self.mt_thread.start()

#         # --- LAN server ---
#         self.lan = LANAudioServer(lan_cfg) if lan_cfg.enable else None
#         self.lan_cfg = lan_cfg

#     # ---------- Hotkeys ----------
#     def _toggle_mute(self):
#         self.tts_muted = not self.tts_muted
#         state = "muted" if self.tts_muted else "unmuted"
#         console.print(f"[magenta]Broadcast: [bold]{state}[/bold].[/magenta]")

#     def _cycle_voice(self):
#         if not isinstance(self.tts, SileroTTS):
#             console.print("[yellow]Voice cycling (F2) is available only with Silero TTS.[/yellow]")
#             return
#         self.voice_idx = (self.voice_idx + 1) % len(self.voice_cycle)
#         new_voice = self.voice_cycle[self.voice_idx]
#         self.tts.speaker = new_voice
#         console.print(f"[magenta]TTS voice → [bold]{new_voice}[/bold][/magenta]")

#     def _setup_hotkeys(self):
#         if keyboard is None:
#             console.print("[yellow]Hotkeys disabled (python 'keyboard' module not available).[/yellow]")
#             return
#         keyboard.add_hotkey("f1", self._toggle_mute)
#         keyboard.add_hotkey("f2", self._cycle_voice)
#         console.print("[dim]Hotkeys: F1 = mute/unmute broadcast • F2 = cycle voice (aidar→eugene→nikolay→baya→xenia)[/dim]")

#     # ---------- Helpers ----------
#     @staticmethod
#     def _is_silent(x: np.ndarray, thresh_db: float = -45.0) -> bool:
#         rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
#         db = 20.0 * np.log10(rms)
#         return db < thresh_db

#     def audio_callback(self, indata, frames, time_info, status):
#         if status:
#             console.log(f"[yellow]Audio status: {status}[/]")
#         audio = np.clip(indata[:, 0], -1.0, 1.0)
#         pcm16 = (audio * 32768.0).astype(np.int16).tobytes()
#         seg = self.segmenter.process_block(pcm16)
#         if seg is not None:
#             self.q.put(seg)

#     def _mt_worker(self):
#         while not self.stop_event.is_set():
#             try:
#                 src_text, asr_rt, start, mid = self.translate_q.get(timeout=0.1)
#             except queue.Empty:
#                 continue
#             t0 = time.time()
#             tgt_text = self.translator.translate(src_text)
#             tgt_text = self.glossary.apply(tgt_text)
#             t1 = time.time()
#             self.print_q.put((src_text, tgt_text, asr_rt, mid - start - asr_rt, t1 - mid, t1 - start))

#     def _speak(self, text: str):
#         if self.tts_muted or not text.strip():
#             return
#         try:
#             audio, sr = self.tts.synthesize(text)
#             # Broadcast only (no local playback)
#             if self.lan and audio.size:
#                 self.lan.send(audio, sr)
#         except Exception as e:
#             console.print(f"[red]TTS error:[/red] {e}")

#     def run(self):
#         ac = self.audio_cfg
#         sd.default.samplerate = ac.samplerate
#         sd.default.channels = ac.channels
#         sd.default.latency = 'low'
#         if ac.device is not None:
#             sd.default.device = ac.device

#         self._setup_hotkeys()

#         ip = guess_lan_ip()
#         console.print(Panel.fit(
#             "[bold green]Starting live translation[/bold green]\n"
#             "Broadcast-only: your machine stays silent.\n"
#             "Open [b]http://%s:%d[/b] on devices in the same Wi-Fi and press Start.\n"
#             "Press [b]Ctrl+C[/b] to stop.\n"
#             "[dim]F1: mute/unmute broadcast • F2: cycle voice[/dim]" % (ip, self.lan_cfg.port),
#             box=box.ROUNDED
#         ))

#         with sd.InputStream(callback=self.audio_callback, dtype="float32", blocksize=self.segmenter.block_size):
#             try:
#                 while not self.stop_event.is_set():
#                     # Drain completed translations for display / TTS
#                     try:
#                         while True:
#                             src, tgt, asr_rt, wait_gap, mt_rt, total = self.print_q.get_nowait()
#                             self._print_result(src, tgt, asr_rt, wait_gap, mt_rt, total)
#                             self._speak(tgt)
#                     except queue.Empty:
#                         pass

#                     # Get a new audio segment if available
#                     try:
#                         segment = self.q.get(timeout=0.05)
#                     except queue.Empty:
#                         continue

#                     if LiveTranslatorApp._is_silent(segment, -45.0):
#                         continue

#                     start = time.time()
#                     src_text, asr_rt = self.transcriber.transcribe(segment, self.audio_cfg.samplerate)
#                     mid = time.time()

#                     # Guards: skip junk, duplicates, non-English
#                     if not src_text or len(src_text.strip()) < 3 or len(src_text.split()) < 2:
#                         continue
#                     if not mostly_latin(src_text):
#                         continue
#                     core = src_text.strip().strip(".?!,:;—-")
#                     if not core:
#                         continue
#                     # Deduplicate exact repeats (5s window)
#                     if not hasattr(self, "_last_src"):
#                         self._last_src = ("", 0.0)
#                     last_text, last_t = self._last_src
#                     now = time.time()
#                     norm = core.lower()
#                     if norm == last_text and (now - last_t) < 5.0:
#                         continue
#                     self._last_src = (norm, now)

#                     # Normalize scripture refs
#                     src_text = fix_refs(src_text)

#                     # Send to MT worker (non-blocking)
#                     try:
#                         self.translate_q.put_nowait((src_text, asr_rt, start, mid))
#                     except queue.Full:
#                         # If MT is backed up, drop this chunk rather than stalling ASR
#                         pass

#             except KeyboardInterrupt:
#                 console.print("\n[cyan]Stopping...[/cyan]")
#                 self.stop_event.set()
#             finally:
#                 # Clean up hotkeys on exit
#                 try:
#                     if keyboard is not None:
#                         keyboard.clear_all_hotkeys()
#                 except Exception:
#                     pass
#                 # Give MT thread a moment to finish
#                 try:
#                     self.mt_thread.join(timeout=1.0)
#                 except Exception:
#                     pass

#     def _print_result(self, src: str, tgt: str, asr_rt: float, wait_gap: float, mt_rt: float, total: float):
#         if not src.strip():
#             return
#         tbl = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
#         tbl.add_row("[bold]Source[/bold]", src.strip())
#         tbl.add_row("[bold]Translation[/bold]", f"[green]{tgt.strip()}[/green]")
#         timing = f"ASR {asr_rt*1000:.0f} ms • MT {mt_rt*1000:.0f} ms • Total {total*1000:.0f} ms"
#         console.print(Panel(tbl, title=timing, border_style="blue"))

# # --------------------- Utils ---------------------
# def guess_lan_ip() -> str:
#     """Best-effort LAN IP for user hinting."""
#     ip = "127.0.0.1"
#     s = None
#     try:
#         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         s.connect(("10.255.255.255", 1))  # doesn't need to be reachable
#         ip = s.getsockname()[0]
#     except Exception:
#         try:
#             ip = socket.gethostbyname(socket.gethostname())
#         except Exception:
#             pass
#     finally:
#         try:
#             if s:
#                 s.close()
#         except Exception:
#             pass
#     return ip

# # --------------------- CLI ---------------------
# def parse_args():
#     p = argparse.ArgumentParser(description="Live offline EN→RU speech translation (broadcast-only over LAN)")
#     p.add_argument("--tts-engine", choices=["silero", "piper"], default="silero", help="Choose TTS engine")
#     p.add_argument("--tts-voice", default="aidar", help="TTS voice (silero: aidar/eugene/nikolay/baya/xenia)")
#     p.add_argument("--asr-lang", default="en", help="Force ASR language ('en' or 'None' to autodetect)")
#     p.add_argument("--whisper", default="small", help="Whisper size: tiny/base/small/medium/large-v3")
#     p.add_argument("--device", default="auto", help="ASR device: auto|cuda|cpu")
#     p.add_argument("--compute-type", choices=["float16", "int8", "int8_float16", "float32"], help="Whisper compute type override")
#     p.add_argument("--beam", type=int, default=1, help="ASR beam size (1 = greedy fastest)")
#     p.add_argument("--vad", action="store_true", help="Also enable Whisper's internal VAD")
#     p.add_argument("--block-ms", type=int, choices=[10, 20, 30], default=20, help="Audio block size for VAD (10/20/30 ms)")
#     p.add_argument("--silence-ms", type=int, default=320, help="Silence to end segment")
#     p.add_argument("--vad-aggr", type=int, default=3, help="WebRTC VAD aggressiveness 0-3")
#     p.add_argument("--glossary", type=str, default="glossary.ru.json", help="Glossary JSON path")
#     p.add_argument("--mt-model", type=str, default="Helsinki-NLP/opus-mt-en-ru", help="MT model name (Marian or NLLB)")
#     p.add_argument("--no-tts", action="store_true", help="Disable TTS (no broadcast)")
#     p.add_argument("--piper-exe", type=str, default=os.getenv("PIPER_EXE"))
#     p.add_argument("--piper-voice", type=str, default=os.getenv("VOICE_PATH"))
#     p.add_argument("--lan-host", default="0.0.0.0", help="LAN bind host (default 0.0.0.0)")
#     p.add_argument("--lan-port", type=int, default=8765, help="LAN HTTP/WebSocket port")
#     p.add_argument("--no-lan", action="store_true", help="Disable LAN audio server")
#     return p.parse_args()

# def main():
#     args = parse_args()
#     a_cfg = AudioConfig(block_ms=args.block_ms, max_silence_ms=args.silence_ms, vad_aggressiveness=args.vad_aggr)
#     w_cfg = WhisperConfig(model_size=args.whisper, device=args.device, beam_size=args.beam, vad_filter=args.vad,
#                           compute_type=args.compute_type,
#                           language=None if (args.asr_lang is None or str(args.asr_lang).lower() == "none") else args.asr_lang)
#     m_cfg = MTConfig(model_name=args.mt_model, src_lang=nllb_code("en"), tgt_lang=nllb_code("ru"))
#     p_cfg = PiperConfig(enabled=not args.no_tts, exe_path=args.piper_exe, voice_path=args.piper_voice,
#                         engine=args.tts_engine, tts_voice=args.tts_voice)
#     lan_cfg = LANConfig(host=args.lan_host, port=args.lan_port, enable=not args.no_lan)
#     app = LiveTranslatorApp(a_cfg, w_cfg, m_cfg, p_cfg, Path(args.glossary), lan_cfg)
#     app.run()

# if __name__ == "__main__":
#     main()















# from __future__ import annotations
# import os, sys, re, time, json, queue, argparse, threading, socket
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Optional, List, Tuple

# # ---------- Environment speedups (set BEFORE heavy imports) ----------
# os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))
# os.environ.setdefault("MKL_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# # Windows: add CUDA bin dirs if present so DLLs load when you switch to --device cuda
# if sys.platform.startswith("win"):
#     for cuda_ver in ("v12.4", "v13.0"):  # prefer 12.4 first
#         bin_dir = fr"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\{cuda_ver}\\bin"
#         if os.path.isdir(bin_dir):
#             try:
#                 os.add_dll_directory(bin_dir)
#             except Exception:
#                 pass

# import numpy as np
# import sounddevice as sd
# import webrtcvad
# from rich.console import Console
# from rich.table import Table
# from rich.panel import Panel
# from rich import box
# from faster_whisper import WhisperModel
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# # --- Optional: global hotkeys (F1/F2) ---
# try:
#     import keyboard  # pip install keyboard
# except Exception:
#     keyboard = None  # hotkeys will be disabled if unavailable

# console = Console()

# # --------------------- Language utilities ---------------------
# NLLB_LANG = {"en":"eng_Latn","ru":"rus_Cyrl","es":"spa_Latn","fr":"fra_Latn","de":"deu_Latn"}

# def nllb_code(lang: str) -> str:
#     return NLLB_LANG.get(lang.lower(), "eng_Latn")

# # --------------------- Config ---------------------
# @dataclass
# class AudioConfig:
#     samplerate: int = 16000
#     channels: int = 1
#     block_ms: int = 20
#     vad_aggressiveness: int = 3
#     max_silence_ms: int = 320
#     device: Optional[int] = None

# @dataclass
# class WhisperConfig:
#     model_size: str = "small"    # tiny/base/small/medium/large-v3
#     device: str = "auto"         # cuda|cpu|auto
#     compute_type: Optional[str] = None  # float16|int8|int8_float16
#     beam_size: int = 1
#     vad_filter: bool = False
#     language: Optional[str] = "en"  # force English ASR by default
#     initial_prompt: Optional[str] = (
#         "Church sermon, biblical terms, Matthew, Mark, Luke, John, Paul, Holy Spirit, Lord's Prayer"
#     )

# @dataclass
# class MTConfig:
#     model_name: str = "Helsinki-NLP/opus-mt-en-ru"  # fast, accurate on CPU
#     src_lang: str = "eng_Latn"
#     tgt_lang: str = "rus_Cyrl"
#     max_new_tokens: int = 128

# @dataclass
# class PiperConfig:
#     enabled: bool = False
#     exe_path: Optional[str] = None
#     voice_path: Optional[str] = None
#     engine: str = "silero"            # "silero" or "piper"
#     tts_voice: str = "aidar"          # silero voices: aidar/eugene/nikolay/baya/xenia

# # --------------------- LAN Audio Server Config ---------------------
# @dataclass
# class LANConfig:
#     host: str = "0.0.0.0"   # listen on all interfaces
#     port: int = 8765        # http://<server-ip>:8765
#     enable: bool = True

# # --------------------- Helpers ---------------------
# _LATIN_RATIO_MIN = 0.85

# def mostly_latin(s: str) -> bool:
#     if not s: return False
#     letters = [ch for ch in s if ch.isalpha()]
#     if not letters: return False
#     latin = sum('A' <= ch <= 'Z' or 'a' <= ch <= 'z' for ch in letters)
#     return (latin / max(1, len(letters))) >= _LATIN_RATIO_MIN

# REF_BOOKS = {
#     r"(Matt(?:hew)?\.?)": "Matthew", r"(Mark\.?)": "Mark", r"(Luke\.?)": "Luke",
#     r"(John\.?)": "John", r"(Rom(?:ans)?\.?)": "Romans"
# }
# _ref_patterns = [ (re.compile(fr"\\b{pat}\\s*([0-9]+)\\s*([0-9]+)\\b", re.I), book) for pat, book in REF_BOOKS.items() ]

# def fix_refs(s: str) -> str:
#     out = s
#     for pat, book in _ref_patterns:
#         out = pat.sub(lambda m: f"{book} {m.group(1)}:{m.group(2)}", out)
#     return out

# # --------------------- Audio + VAD capture ---------------------
# class VADSegmenter:
#     def __init__(self, cfg: AudioConfig):
#         self.cfg = cfg
#         self.vad = webrtcvad.Vad(cfg.vad_aggressiveness)
#         self.block_size = int(cfg.samplerate * cfg.block_ms / 1000)
#         self.silence_blocks = cfg.max_silence_ms // cfg.block_ms
#         self.frames: List[bytes] = []
#         self.silent_count = 0

#     def process_block(self, pcm16: bytes) -> Optional[np.ndarray]:
#         is_speech = self.vad.is_speech(pcm16, sample_rate=self.cfg.samplerate)
#         if is_speech:
#             self.frames.append(pcm16)
#             self.silent_count = 0
#             return None
#         else:
#             if self.frames:
#                 self.silent_count += 1
#                 self.frames.append(pcm16)
#                 if self.silent_count >= self.silence_blocks:
#                     chunk = b"".join(self.frames)
#                     self.frames = []
#                     self.silent_count = 0
#                     audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
#                     return audio
#             return None

# # --------------------- ASR (faster-whisper) ---------------------
# class Transcriber:
#     def __init__(self, cfg: WhisperConfig):
#         device = cfg.device
#         if device == "auto":
#             device = "cuda" if _has_cuda() else "cpu"
#         compute_type = cfg.compute_type or ("float16" if device == "cuda" else "int8")
#         console.print(f"[cyan]Loading Whisper '{cfg.model_size}' on {device} ({compute_type})...[/cyan]")
#         self.model = WhisperModel(
#             cfg.model_size,
#             device=device,
#             compute_type=compute_type,
#             cpu_threads=max(1, (os.cpu_count() or 4) - 1)
#         )
#         self.cfg = cfg

#     def transcribe(self, audio: np.ndarray, samplerate: int) -> Tuple[str, float]:
#         start = time.time()
#         segments, info = self.model.transcribe(
#             audio,
#             language=self.cfg.language,
#             beam_size=self.cfg.beam_size,
#             vad_filter=self.cfg.vad_filter,
#             word_timestamps=False,
#             condition_on_previous_text=False,
#             temperature=0.0,
#             no_speech_threshold=0.6,
#             log_prob_threshold=-0.5,
#             initial_prompt=self.cfg.initial_prompt,
#             vad_filter=self.cfg.vad_filter,
#         )
#         text = "".join(seg.text for seg in segments).strip()
#         return text, (time.time() - start)

# def _has_cuda() -> bool:
#     return os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "") or os.name == "nt" or Path("/proc/driver/nvidia/version").exists()

# # --------------------- Translator (Marian or NLLB) ---------------------
# class Translator:
#     def __init__(self, cfg: MTConfig):
#         self.cfg = cfg
#         console.print(f"[cyan]Loading translator '{cfg.model_name}'...[/cyan]")
#         self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name,use_safetensors=True)

#         if "opus-mt-" in cfg.model_name:
#             task = "translation_en_to_ru"
#             self.pipe = pipeline(
#                 task,
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 max_new_tokens=min(128, self.cfg.max_new_tokens),
#                 truncation=True,
#                 num_beams=1,
#                 no_repeat_ngram_size=3,
#                 length_penalty=1.0,
#             )
#         else:
#             self.pipe = pipeline(
#                 "translation",
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 src_lang=cfg.src_lang,
#                 tgt_lang=cfg.tgt_lang,
#                 max_new_tokens=self.cfg.max_new_tokens,
#                 truncation=True,
#             )

#     def translate(self, text: str) -> str:
#         if not text:
#             return ""
#         return self.pipe(text)[0]["translation_text"]

# # --------------------- Silero TTS (CPU, offline) ---------------------
# class SileroTTS:
#     """
#     CPU-friendly Russian TTS using Silero (offline).
#     Voices: 'aidar' (male), 'eugene' (male), 'nikolay' (male), 'baya' (female), 'xenia' (female)
#     """
#     def __init__(self, speaker: str = "aidar", sample_rate: int = 48000):
#         import torch
#         self.torch = torch
#         self.sample_rate = int(sample_rate)
#         self.speaker = speaker
#         # Load the small RU multi-speaker Silero model (CPU)
#         self.model, _ = torch.hub.load(
#             'snakers4/silero-models', 'silero_tts',
#             language='ru', speaker='v3_1_ru', trust_repo=True
#         )

#     def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
#         if not text or not text.strip():
#             return np.zeros(0, dtype=np.float32), self.sample_rate
#         with self.torch.no_grad():
#             audio = self.model.apply_tts(text=text, speaker=self.speaker, sample_rate=self.sample_rate)
#         return np.array(audio, dtype=np.float32), self.sample_rate

# # --------------------- Optional Piper TTS ---------------------
# class PiperTTS:
#     def __init__(self, cfg: PiperConfig):
#         self.enabled = bool(cfg.enabled and cfg.exe_path and cfg.voice_path and Path(cfg.exe_path).exists() and Path(cfg.voice_path).exists())
#         self.exe = cfg.exe_path; self.voice = cfg.voice_path
#     def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
#         if not (self.enabled and text):
#             return np.zeros(0, dtype=np.float32), 22050
#         import subprocess, tempfile, soundfile as sf
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
#             wav_path = wf.name
#         cmd = [self.exe, "-m", self.voice, "-f", wav_path, "-q"]
#         p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         p.communicate(input=text.encode("utf-8"), timeout=30)
#         data, sr = sf.read(wav_path, dtype="float32")
#         return data.astype(np.float32, copy=False), int(sr)

# # --------------------- Glossary ---------------------
# class Glossary:
#     def __init__(self, path: Path, tgt_lang: str):
#         self.map = {}
#         if path.exists():
#             try:
#                 data = json.loads(path.read_text(encoding="utf-8"))
#                 if data.get("target_lang") in (tgt_lang, None):
#                     self.map = data.get("replacements", {})
#             except Exception as e:
#                 console.print(f"[yellow]Failed to load glossary ({e}). Continuing without it.[/yellow]")
#     def apply(self, text: str) -> str:
#         out = text
#         for k, v in self.map.items():
#             out = out.replace(k, v)
#         return out

# # --------------------- LAN audio server (WebSocket) ---------------------
# class LANAudioServer:
#     """
#     Simple offline LAN broadcaster.
#     - Serves / (HTML player) and /ws (WebSocket)
#     - Broadcasts float32 PCM frames with a tiny binary header:
#         b'PCM0' + uint32_le(sample_rate) + uint32_le(n_samples) + float32[n_samples]
#     """
#     HTML_PAGE = r"""<!doctype html>
# <html lang="en"><meta charset="utf-8"><title>LAN TTS</title>
# <style>body{font-family:system-ui;margin:2rem}button{padding:.6rem 1rem}#log{margin-top:2rem;white-space:pre-wrap;background:#f0f0f0;padding:1rem;max-height:300px;overflow-y:auto;border:1px solid #ccc;border-radius:4px;font-size:0.9rem;}</style>
# <h2>LAN TTS Audio</h2>
# <p>Press Start (a user gesture is required for audio). This works offline on the same Wi‑Fi/LAN.</p>
# <button id="btn">Start</button>
# <p id="status">Idle</p>
# <div><label><input type="checkbox" id="logToggle"> Enable debug log</label></div>
# <div id="log" style="display:none;"></div>
# <script>
# let ctx, node, queue=[], cur=null, idx=0, playing=false, ws=null, logEnabled=false;

# function log(msg){
#   if(logEnabled){
#     let logArea=document.getElementById('log');
#     logArea.textContent+=msg+"\n";
#     logArea.scrollTop=logArea.scrollHeight;
#   }
# }

# function setupAudio(){
#   if(ctx) return;
#   try{
#     ctx = new (window.AudioContext||window.webkitAudioContext)({ sampleRate: 48000 });
#     log('AudioContext created with sampleRate: '+ctx.sampleRate);
#   }catch(e){
#     ctx = new (window.AudioContext||window.webkitAudioContext)();
#     log('AudioContext created with default sampleRate');
#   }
#   node = ctx.createScriptProcessor(2048, 0, 1);
#   node.onaudioprocess = e=>{
#     let out = e.outputBuffer.getChannelData(0);
#     for(let i=0;i<out.length;i++){
#       if(!cur || idx>=cur.length){ cur = queue.length ? queue.shift() : null; idx = 0; }
#       out[i] = (cur ? cur[idx++] : 0);
#     }
#   };
#   node.connect(ctx.destination);
# }

# function resampleFloat32(data, srcRate, dstRate){
#   if(srcRate===dstRate) return data;
#   const ratio = srcRate/dstRate;
#   const newLen = Math.max(1, Math.round(data.length/ratio));
#   const out=new Float32Array(newLen);
#   for(let i=0;i<newLen;i++){
#     const idx = i*ratio;
#     const i0 = Math.floor(idx);
#     const frac = idx - i0;
#     const s0 = data[i0]||0;
#     const s1 = data[i0+1]||s0;
#     out[i] = s0+frac*(s1-s0);
#   }
#   return out;
# }

# function startWS(){
#   if(ws) return;
#   const proto = location.protocol === 'https:' ? 'wss' : 'ws';
#   ws = new WebSocket(proto + '://' + location.host + '/ws');
#   ws.binaryType = 'arraybuffer';
#   ws.onopen = ()=> { document.getElementById('status').innerText='Connected. Waiting for audio…'; log('WebSocket connected'); };
#   ws.onclose = ()=> { document.getElementById('status').innerText='Disconnected.'; log('WebSocket disconnected'); };
#   ws.onerror = e=> { document.getElementById('status').innerText='Error.'; log('WebSocket error: '+e.message); };
#   ws.onmessage = ev=>{
#     try {
#       const dv = new DataView(ev.data);
#       if(dv.getUint8(0)!==0x50 || dv.getUint8(1)!==0x43 || dv.getUint8(2)!==0x4D || dv.getUint8(3)!==0x30){
#         log('Invalid header received');
#         return;
#       }
#       const sr = dv.getUint32(4,true);
#       const f32 = new Float32Array(ev.data, 12);
#       const data = resampleFloat32(f32, sr, ctx.sampleRate);
#       queue.push(data);
#       if(!playing){ playing=true; document.getElementById('status').innerText=`Playing @ ${ctx.sampleRate} Hz…`; }
#       log('Received audio chunk with sampleRate: ' + sr + ', length: ' + f32.length);
#     } catch(ex) {
#       log('Error processing websocket message: ' + ex.message);
#     }
#   };
# }

# document.getElementById('btn').onclick = ()=>{
#   setupAudio();
#   startWS();
#   document.getElementById('btn').disabled=true;
# };

# document.getElementById('logToggle').onchange = e => {
#   logEnabled = e.target.checked;
#   document.getElementById('log').style.display = logEnabled ? 'block' : 'none';
#   if(logEnabled) log('Debug log enabled');
# }
# </script>
# </html>

# """

#     def __init__(self, cfg: LANConfig):
#         self.cfg = cfg
#         self._q: "queue.Queue[Tuple[np.ndarray,int]]" = queue.Queue(maxsize=32)
#         self._thread = threading.Thread(target=self._run, daemon=True)
#         self._thread.start()

#     def send(self, samples: np.ndarray, sample_rate: int):
#         """Queue one chunk (float32 mono). Non-blocking."""
#         try:
#             self._q.put_nowait((samples.astype(np.float32, copy=False), int(sample_rate)))
#         except queue.Full:
#             pass

#     def _run(self):
#         import asyncio, struct
#         from contextlib import suppress
#         try:
#             import aiohttp
#             from aiohttp import web
#         except Exception as e:
#             console.print(f"[red]aiohttp not installed: {e}. LAN broadcast disabled.[/red]")
#             return

#         # On Windows, ensure a selector loop in this background thread
#         try:
#             if sys.platform.startswith("win"):
#                 asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#         except Exception:
#             pass

#         async def index(_):
#             return web.Response(text=self.HTML_PAGE, content_type="text/html")

#         websockets: "set[web.WebSocketResponse]" = set()

#         async def ws_handler(request):
#             ws = web.WebSocketResponse()
#             await ws.prepare(request)
#             websockets.add(ws)
#             try:
#                 async for _ in ws:
#                     pass
#             finally:
#                 websockets.discard(ws)
#             return ws

#         async def broadcaster():
#             loop = asyncio.get_running_loop()
#             while True:
#                 samples, sr = await loop.run_in_executor(None, self._q.get)
#                 header = struct.pack("<4sII", b"PCM0", sr, samples.shape[0])
#                 payload = header + samples.tobytes(order="C")
#                 dead = []
#                 for ws in list(websockets):
#                     try:
#                         await ws.send_bytes(payload)
#                     except Exception:
#                         dead.append(ws)
#                 for d in dead:
#                     websockets.discard(d)

#         async def on_startup(app):
#             app['broadcaster_task'] = asyncio.create_task(broadcaster())
#             console.print(f"[green]LAN server listening on http://{guess_lan_ip()}:{self.cfg.port}[/green]")

#         async def on_cleanup(app):
#             task = app.get('broadcaster_task')
#             if task:
#                 task.cancel()
#                 with suppress(asyncio.CancelledError):
#                     await task

#         app = web.Application()
#         app.router.add_get("/", index)
#         app.router.add_get("/ws", ws_handler)
#         app.on_startup.append(on_startup)
#         app.on_cleanup.append(on_cleanup)

#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         runner = web.AppRunner(app)
#         loop.run_until_complete(runner.setup())
#         site = web.TCPSite(runner, self.cfg.host, self.cfg.port)
#         try:
#             loop.run_until_complete(site.start())
#         except OSError as e:
#             console.print(f"[red]Failed to bind {self.cfg.host}:{self.cfg.port} — {e}[/red]")
#             return
#         try:
#             loop.run_forever()
#         finally:
#             loop.run_until_complete(runner.cleanup())
#             loop.close()

# # --------------------- Main App with concurrency ---------------------
# class LiveTranslatorApp:
#     def __init__(self, a_cfg: AudioConfig, w_cfg: WhisperConfig, m_cfg: MTConfig, p_cfg: PiperConfig, glossary_path: Path, lan_cfg: LANConfig):
#         self.audio_cfg = a_cfg
#         self.segmenter = VADSegmenter(a_cfg)
#         self.transcriber = Transcriber(w_cfg)
#         self.translator = Translator(m_cfg)
#         self.glossary = Glossary(glossary_path, m_cfg.tgt_lang)

#         # --- TTS selection (Silero default) ---
#         if p_cfg.engine == "silero" and not p_cfg.enabled is False:
#             self.tts = SileroTTS(speaker=p_cfg.tts_voice)
#         elif p_cfg.engine == "piper" and not p_cfg.enabled is False:
#             self.tts = PiperTTS(p_cfg)
#         else:
#             class _NoTTS:
#                 def synthesize(self, text: str): return (np.zeros(0, dtype=np.float32), 24000)
#             self.tts = _NoTTS()

#         # --- Hotkey-related state ---
#         self.tts_muted = False  # controls broadcast now
#         self.voice_cycle = ["aidar", "eugene", "nikolay"]
#         if isinstance(self.tts, SileroTTS) and getattr(self.tts, "speaker", None) in self.voice_cycle:
#             self.voice_idx = self.voice_cycle.index(self.tts.speaker)
#         else:
#             self.voice_idx = 0

#         self.q = queue.Queue()
#         self.translate_q: queue.Queue = queue.Queue(maxsize=4)
#         self.print_q: queue.Queue = queue.Queue(maxsize=8)
#         self.stop_event = threading.Event()
#         # Warm-up (removes 1st-use spikes)
#         _ = self.translator.translate("Hello")
#         dummy = np.zeros(int(self.audio_cfg.samplerate * 0.5), dtype=np.float32)
#         _ = self.transcriber.transcribe(dummy, self.audio_cfg.samplerate)
#         # MT worker
#         self.mt_thread = threading.Thread(target=self._mt_worker, daemon=True)
#         self.mt_thread.start()

#         # --- LAN server ---
#         self.lan = LANAudioServer(lan_cfg) if lan_cfg.enable else None
#         self.lan_cfg = lan_cfg

#     # ---------- Hotkeys ----------
#     def _toggle_mute(self):
#         self.tts_muted = not self.tts_muted
#         state = "muted" if self.tts_muted else "unmuted"
#         console.print(f"[magenta]Broadcast: [bold]{state}[/bold].[/magenta]")

#     def _cycle_voice(self):
#         if not isinstance(self.tts, SileroTTS):
#             console.print("[yellow]Voice cycling (F2) is available only with Silero TTS.[/yellow]")
#             return
#         self.voice_idx = (self.voice_idx + 1) % len(self.voice_cycle)
#         new_voice = self.voice_cycle[self.voice_idx]
#         self.tts.speaker = new_voice
#         console.print(f"[magenta]TTS voice → [bold]{new_voice}[/bold][/magenta]")

#     def _setup_hotkeys(self):
#         if keyboard is None:
#             console.print("[yellow]Hotkeys disabled (python 'keyboard' module not available).[/yellow]")
#             return
#         keyboard.add_hotkey("f1", self._toggle_mute)
#         keyboard.add_hotkey("f2", self._cycle_voice)
#         console.print("[dim]Hotkeys: F1 = mute/unmute broadcast • F2 = cycle male voice (aidar→eugene→nikolay)[/dim]")

#     # ---------- Helpers ----------
#     @staticmethod
#     def _is_silent(x: np.ndarray, thresh_db: float = -45.0) -> bool:
#         rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
#         db = 20.0 * np.log10(rms)
#         return db < thresh_db

#     def audio_callback(self, indata, frames, time_info, status):
#         if status:
#             console.log(f"[yellow]Audio status: {status}[/]")
#         audio = np.clip(indata[:, 0], -1.0, 1.0)
#         pcm16 = (audio * 32768.0).astype(np.int16).tobytes()
#         seg = self.segmenter.process_block(pcm16)
#         if seg is not None:
#             self.q.put(seg)

#     def _mt_worker(self):
#         while not self.stop_event.is_set():
#             try:
#                 src_text, asr_rt, start, mid = self.translate_q.get(timeout=0.1)
#             except queue.Empty:
#                 continue
#             t0 = time.time()
#             tgt_text = self.translator.translate(src_text)
#             tgt_text = self.glossary.apply(tgt_text)
#             t1 = time.time()
#             self.print_q.put((src_text, tgt_text, asr_rt, mid - start - asr_rt, t1 - mid, t1 - start))

#     def _speak(self, text: str):
#         if self.tts_muted or not text.strip():
#             return
#         try:
#             audio, sr = self.tts.synthesize(text)
#             # Broadcast only (no local playback)
#             if self.lan and audio.size:
#                 self.lan.send(audio, sr)
#         except Exception as e:
#             console.print(f"[red]TTS error:[/red] {e}")

#     def run(self):
#         ac = self.audio_cfg
#         sd.default.samplerate = ac.samplerate
#         sd.default.channels = ac.channels
#         sd.default.latency = 'low'
#         if ac.device is not None:
#             sd.default.device = ac.device

#         self._setup_hotkeys()

#         ip = guess_lan_ip()
#         console.print(Panel.fit(
#             "[bold green]Starting live translation[/bold green]\n"
#             "Broadcast-only: your machine stays silent.\n"
#             "Open [b]http://%s:%d[/b] on devices in the same Wi‑Fi and press Start.\n"
#             "Press [b]Ctrl+C[/b] to stop.\n"
#             "[dim]F1: mute/unmute broadcast • F2: cycle voice[/dim]" % (ip, self.lan_cfg.port),
#             box=box.ROUNDED
#         ))

#         with sd.InputStream(callback=self.audio_callback, dtype="float32", blocksize=self.segmenter.block_size):
#             try:
#                 while not self.stop_event.is_set():
#                     # Drain completed translations for display / TTS
#                     try:
#                         while True:
#                             src, tgt, asr_rt, wait_gap, mt_rt, total = self.print_q.get_nowait()
#                             self._print_result(src, tgt, asr_rt, wait_gap, mt_rt, total)
#                             self._speak(tgt)
#                     except queue.Empty:
#                         pass

#                     # Get a new audio segment if available
#                     try:
#                         segment = self.q.get(timeout=0.05)
#                     except queue.Empty:
#                         continue

#                     if LiveTranslatorApp._is_silent(segment, -45.0):
#                         continue

#                     start = time.time()
#                     src_text, asr_rt = self.transcriber.transcribe(segment, self.audio_cfg.samplerate)
#                     mid = time.time()

#                     # Guards: skip junk, duplicates, non-English
#                     if not src_text or len(src_text.strip()) < 3 or len(src_text.split()) < 2:
#                         continue
#                     if not mostly_latin(src_text):
#                         continue
#                     core = src_text.strip().strip(".?!,:;—-")
#                     if not core:
#                         continue
#                     # Deduplicate exact repeats (5s window)
#                     if not hasattr(self, "_last_src"):
#                         self._last_src = ("", 0.0)
#                     last_text, last_t = self._last_src
#                     now = time.time()
#                     norm = core.lower()
#                     if norm == last_text and (now - last_t) < 5.0:
#                         continue
#                     self._last_src = (norm, now)

#                     # Normalize scripture refs
#                     src_text = fix_refs(src_text)

#                     # Send to MT worker (non-blocking)
#                     try:
#                         self.translate_q.put_nowait((src_text, asr_rt, start, mid))
#                     except queue.Full:
#                         # If MT is backed up, drop this chunk rather than stalling ASR
#                         pass

#             except KeyboardInterrupt:
#                 console.print("\n[cyan]Stopping...[/cyan]")
#                 self.stop_event.set()
#             finally:
#                 # Clean up hotkeys on exit
#                 try:
#                     if keyboard is not None:
#                         keyboard.clear_all_hotkeys()
#                 except Exception:
#                     pass

#     def _print_result(self, src: str, tgt: str, asr_rt: float, wait_gap: float, mt_rt: float, total: float):
#         if not src.strip():
#             return
#         tbl = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0,1))
#         tbl.add_row("[bold]Source[/bold]", src.strip())
#         tbl.add_row("[bold]Translation[/bold]", f"[green]{tgt.strip()}[/green]")
#         timing = f"ASR {asr_rt*1000:.0f} ms • MT {mt_rt*1000:.0f} ms • Total {total*1000:.0f} ms"
#         console.print(Panel(tbl, title=timing, border_style="blue"))

# # --------------------- Utils ---------------------
# def guess_lan_ip() -> str:
#     """Best-effort LAN IP for user hinting."""
#     ip = "127.0.0.1"
#     try:
#         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         s.connect(("10.255.255.255", 1))  # doesn't need to be reachable
#         ip = s.getsockname()[0]
#     except Exception:
#         try:
#             ip = socket.gethostbyname(socket.gethostname())
#         except Exception:
#             pass
#     finally:
#         try:
#             s.close()
#         except Exception:
#             pass
#     return ip

# # --------------------- CLI ---------------------
# def parse_args():
#     p = argparse.ArgumentParser(description="Live offline EN→RU speech translation (broadcast-only over LAN)")
#     p.add_argument("--tts-engine", choices=["silero", "piper"], default="silero", help="Choose TTS engine")
#     p.add_argument("--tts-voice", default="aidar", help="TTS voice (silero: aidar/eugene/nikolay/baya/xenia)")
#     p.add_argument("--asr-lang", default="en", help="Force ASR language ('en' or 'None' to autodetect)")
#     p.add_argument("--whisper", default="small", help="Whisper size: tiny/base/small/medium/large-v3")
#     p.add_argument("--device", default="auto", help="ASR device: auto|cuda|cpu")
#     p.add_argument("--beam", type=int, default=1, help="ASR beam size (1 = greedy fastest)")
#     p.add_argument("--vad", action="store_true", help="Also enable Whisper's internal VAD")
#     p.add_argument("--block-ms", type=int, default=20, help="Audio block size for VAD (10/20/30 ms)")
#     p.add_argument("--silence-ms", type=int, default=320, help="Silence to end segment")
#     p.add_argument("--vad-aggr", type=int, default=3, help="WebRTC VAD aggressiveness 0-3")
#     p.add_argument("--glossary", type=str, default="glossary.ru.json", help="Glossary JSON path")
#     p.add_argument("--mt-model", type=str, default="Helsinki-NLP/opus-mt-en-ru", help="MT model name (Marian or NLLB)")
#     p.add_argument("--no-tts", action="store_true", help="Disable TTS (no broadcast)")
#     p.add_argument("--piper-exe", type=str, default=os.getenv("PIPER_EXE"))
#     p.add_argument("--piper-voice", type=str, default=os.getenv("VOICE_PATH"))
#     p.add_argument("--lan-host", default="0.0.0.0", help="LAN bind host (default 0.0.0.0)")
#     p.add_argument("--lan-port", type=int, default=8765, help="LAN HTTP/WebSocket port")
#     p.add_argument("--no-lan", action="store_true", help="Disable LAN audio server")
#     return p.parse_args()


# def main():
#     args = parse_args()
#     a_cfg = AudioConfig(block_ms=args.block_ms, max_silence_ms=args.silence_ms, vad_aggressiveness=args.vad_aggr)
#     w_cfg = WhisperConfig(model_size=args.whisper, device=args.device, beam_size=args.beam, vad_filter=args.vad,
#                           language=None if (args.asr_lang is None or str(args.asr_lang).lower()=="none") else args.asr_lang)
#     m_cfg = MTConfig(model_name=args.mt_model, src_lang=nllb_code("en"), tgt_lang=nllb_code("ru"))
#     p_cfg = PiperConfig(enabled=not args.no_tts, exe_path=args.piper_exe, voice_path=args.piper_voice,
#                         engine=args.tts_engine, tts_voice=args.tts_voice)
#     lan_cfg = LANConfig(host=args.lan_host, port=args.lan_port, enable=not args.no_lan)
#     app = LiveTranslatorApp(a_cfg, w_cfg, m_cfg, p_cfg, Path(args.glossary), lan_cfg)
#     app.run()

# if __name__ == "__main__":
#     main()















# #!/usr/bin/env python3
# from __future__ import annotations
# import os, sys, re, time, json, queue, argparse, threading
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Optional, List, Tuple

# # ---------- Environment speedups ----------
# os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))
# os.environ.setdefault("MKL_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# # Windows CUDA DLL path (safe to keep even on CPU)
# if sys.platform.startswith("win"):
#     for cuda_ver in ("v12.4", "v13.0"):
#         bin_dir = fr"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{cuda_ver}\bin"
#         if os.path.isdir(bin_dir):
#             try:
#                 os.add_dll_directory(bin_dir)
#             except Exception:
#                 pass

# # ---------- Imports ----------
# import numpy as np
# import sounddevice as sd
# import webrtcvad
# from rich.console import Console
# from rich.table import Table
# from rich.panel import Panel
# from rich import box
# from faster_whisper import WhisperModel
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# # Hotkeys (optional)
# _KEYBOARD_OK = True
# try:
#     import keyboard  # requires admin on some Windows setups
# except Exception:
#     _KEYBOARD_OK = False

# console = Console()

# # ---------- Language utilities ----------
# NLLB_LANG = {"en":"eng_Latn","ru":"rus_Cyrl","es":"spa_Latn","fr":"fra_Latn","de":"deu_Latn"}
# def nllb_code(lang: str) -> str:
#     return NLLB_LANG.get(lang.lower(), "eng_Latn")

# # ---------- Config ----------
# @dataclass
# class AudioConfig:
#     samplerate: int = 16000
#     channels: int = 1
#     block_ms: int = 10
#     vad_aggressiveness: int = 3
#     max_silence_ms: int = 180
#     device: Optional[int] = None

# @dataclass
# class WhisperConfig:
#     model_size: str = "small"
#     device: str = "auto"         # cuda|cpu|auto
#     compute_type: Optional[str] = None
#     beam_size: int = 1
#     vad_filter: bool = False
#     language: Optional[str] = "en"
#     initial_prompt: Optional[str] = ("Church sermon, biblical terms, Matthew, Mark, Luke, John, "
#                                      "Paul, Holy Spirit, Lord's Prayer, Gospel, Scripture")

# @dataclass
# class MTConfig:
#     model_name: str = "Helsinki-NLP/opus-mt-en-ru"
#     src_lang: str = "eng_Latn"
#     tgt_lang: str = "rus_Cyrl"
#     max_new_tokens: int = 128

# @dataclass
# class PiperConfig:
#     enabled: bool = False
#     exe_path: Optional[str] = None
#     voice_path: Optional[str] = None
#     engine: str = "silero"       # "silero" or "piper"
#     tts_voice: str = "aidar"     # silero voices: aidar/eugene/nikolay/baya/xenia

# # ---------- Helpers ----------
# _LATIN_RATIO_MIN = 0.85
# def mostly_latin(s: str) -> bool:
#     if not s: return False
#     letters = [ch for ch in s if ch.isalpha()]
#     if not letters: return False
#     latin = sum('A' <= ch <= 'Z' or 'a' <= ch <= 'z' for ch in letters)
#     return (latin / max(1, len(letters))) >= _LATIN_RATIO_MIN

# REF_BOOKS = {
#     r"(Matt(?:hew)?\.?)": "Matthew", r"(Mark\.?)": "Mark", r"(Luke\.?)": "Luke",
#     r"(John\.?)": "John", r"(Rom(?:ans)?\.?)": "Romans"
# }
# _ref_patterns = [ (re.compile(fr"\b{pat}\s*([0-9]+)\s*([0-9]+)\b", re.I), book) for pat, book in REF_BOOKS.items() ]
# def fix_refs(s: str) -> str:
#     out = s
#     for pat, book in _ref_patterns:
#         out = pat.sub(lambda m: f"{book} {m.group(1)}:{m.group(2)}", out)
#     return out

# def _has_cuda() -> bool:
#     return os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "") or os.name == "nt" or Path("/proc/driver/nvidia/version").exists()

# def _is_final(text: str) -> bool:
#     return any(text.rstrip().endswith(p) for p in ('.','!','?','…','.”','!”','?”',':',';'))

# # ---------- Streaming Segmenter (time-based flush + overlap) ----------
# class StreamingSegmenter:
#     """
#     Emits audio either:
#       - when VAD detects end-of-speech (silence >= max_silence_ms), OR
#       - every max_chunk_ms even if speech continues (with a tail overlap).
#     """
#     def __init__(self, cfg: AudioConfig, max_chunk_ms: int = 900, overlap_ms: int = 200):
#         self.cfg = cfg
#         self.vad = webrtcvad.Vad(cfg.vad_aggressiveness)
#         self.block_size = int(cfg.samplerate * cfg.block_ms / 1000)
#         self.silence_blocks = cfg.max_silence_ms // cfg.block_ms
#         self.max_blocks = max(1, max_chunk_ms // cfg.block_ms)
#         self.overlap_blocks = max(0, overlap_ms // cfg.block_ms)
#         self.frames: List[bytes] = []
#         self.silent_count = 0
#         self.blocks_since_flush = 0

#     def _emit(self, keep_overlap: bool) -> Optional[np.ndarray]:
#         if not self.frames:
#             return None
#         chunk = b"".join(self.frames)
#         # keep small tail for continuity
#         if keep_overlap and self.overlap_blocks > 0:
#             tail = self.frames[-self.overlap_blocks:]
#         else:
#             tail = []
#         self.frames = tail.copy()
#         self.silent_count = 0
#         self.blocks_since_flush = len(self.frames)
#         audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
#         return audio

#     def process_block(self, pcm16: bytes) -> Optional[np.ndarray]:
#         is_speech = self.vad.is_speech(pcm16, sample_rate=self.cfg.samplerate)
#         self.frames.append(pcm16)
#         self.blocks_since_flush += 1

#         if is_speech:
#             self.silent_count = 0
#         else:
#             self.silent_count += 1
#             if self.silent_count >= self.silence_blocks:
#                 return self._emit(keep_overlap=False)

#         if self.blocks_since_flush >= self.max_blocks:
#             return self._emit(keep_overlap=True)

#         return None

# # ---------- ASR ----------
# class Transcriber:
#     def __init__(self, cfg: WhisperConfig):
#         device = cfg.device if cfg.device != "auto" else ("cuda" if _has_cuda() else "cpu")
#         compute_type = cfg.compute_type or ("float16" if device == "cuda" else "int8")
#         console.print(f"[cyan]Loading Whisper '{cfg.model_size}' on {device} ({compute_type})...[/cyan]")
#         self.model = WhisperModel(
#             cfg.model_size, device=device, compute_type=compute_type,
#             cpu_threads=max(1, (os.cpu_count() or 4) - 1)
#         )
#         self.cfg = cfg

#     def transcribe(self, audio: np.ndarray, samplerate: int) -> Tuple[str, float]:
#         t0 = time.time()
#         segments, _info = self.model.transcribe(
#             audio,
#             language=self.cfg.language,
#             beam_size=self.cfg.beam_size,          # 1 for speed
#             vad_filter=self.cfg.vad_filter,
#             word_timestamps=False,
#             condition_on_previous_text=True,       # continuity across chunks
#             temperature=0.0,                       # stability
#             no_speech_threshold=0.6,
#             log_prob_threshold=-0.5,
#             initial_prompt=self.cfg.initial_prompt,
#         )
#         text = "".join(seg.text for seg in segments).strip()
#         return text, (time.time() - t0)

# # ---------- MT ----------
# class Translator:
#     def __init__(self, cfg: MTConfig):
#         self.cfg = cfg
#         console.print(f"[cyan]Loading translator '{cfg.model_name}'...[/cyan]")
#         self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name, use_safetensors=True)
#         if "opus-mt-" in cfg.model_name:
#             task = "translation_en_to_ru"
#             self.pipe = pipeline(
#                 task, model=self.model, tokenizer=self.tokenizer,
#                 max_new_tokens=min(128, self.cfg.max_new_tokens),
#                 truncation=True, num_beams=1, no_repeat_ngram_size=3, length_penalty=1.0
#             )
#         else:
#             self.pipe = pipeline(
#                 "translation", model=self.model, tokenizer=self.tokenizer,
#                 src_lang=cfg.src_lang, tgt_lang=cfg.tgt_lang,
#                 max_new_tokens=self.cfg.max_new_tokens, truncation=True
#             )
#     def translate(self, text: str) -> str:
#         if not text: return ""
#         return self.pipe(text)[0]["translation_text"]

# # ---------- Silero TTS (CPU, offline) ----------
# class SileroTTS:
#     VOICES = ["aidar","eugene","nikolay","baya","xenia"]
#     def __init__(self, speaker: str = "aidar", sample_rate: int = 24000):
#         import torch
#         self.torch = torch
#         self.sample_rate = sample_rate
#         self.speaker = speaker if speaker in self.VOICES else "aidar"
#         self.model, _ = torch.hub.load('snakers4/silero-models','silero_tts',
#                                        language='ru', speaker='v3_1_ru', trust_repo=True)
#         self._muted = False
#     def set_voice(self, speaker: str):
#         if speaker in self.VOICES:
#             self.speaker = speaker
#     def toggle_mute(self):
#         self._muted = not self._muted
#         return self._muted
#     def speak(self, text: str):
#         if self._muted or not text or not text.strip():
#             return
#         import numpy as np, sounddevice as sd
#         with self.torch.no_grad():
#             audio = self.model.apply_tts(text=text, speaker=self.speaker, sample_rate=self.sample_rate)
#         sd.play(np.array(audio, dtype=np.float32), self.sample_rate); sd.wait()

# # ---------- Piper TTS (optional; not used when silero engine chosen) ----------
# class PiperTTS:
#     def __init__(self, cfg):
#         self.enabled = bool(cfg.enabled and cfg.exe_path and cfg.voice_path and Path(cfg.exe_path).exists() and Path(cfg.voice_path).exists())
#         self.exe, self.voice = cfg.exe_path, cfg.voice_path
#         self._muted = False
#     def set_voice(self, _): pass
#     def toggle_mute(self):
#         self._muted = not self._muted
#         return self._muted
#     def speak(self, text: str):
#         if not self.enabled or self._muted or not text:
#             return
#         import subprocess, tempfile, sounddevice as sd, soundfile as sf
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
#             wav_path = wf.name
#         cmd = [self.exe, "-m", self.voice, "-f", wav_path, "-q"]
#         p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         p.communicate(input=text.encode("utf-8"), timeout=30)
#         data, sr = sf.read(wav_path, dtype="float32"); sd.play(data, sr); sd.wait()

# # ---------- Glossary ----------
# class Glossary:
#     def __init__(self, path: Path, tgt_lang: str):
#         self.map = {}
#         if path.exists():
#             try:
#                 data = json.loads(path.read_text(encoding="utf-8"))
#                 if data.get("target_lang") in (tgt_lang, None):
#                     self.map = data.get("replacements", {})
#             except Exception as e:
#                 console.print(f"[yellow]Failed to load glossary ({e}). Continuing without it.[/yellow]")
#     def apply(self, text: str) -> str:
#         out = text
#         for k, v in self.map.items():
#             out = out.replace(k, v)
#         return out

# # ---------- Main App ----------
# class LiveTranslatorApp:
#     args = None  # will be set from main()

#     def __init__(self, a_cfg: AudioConfig, w_cfg: WhisperConfig, m_cfg: MTConfig, p_cfg: PiperConfig, glossary_path: Path):
#         self.audio_cfg = a_cfg
#         args = LiveTranslatorApp.args

#         # Streaming segmenter defaults
#         max_chunk_ms = getattr(args, "max_chunk_ms", 900)
#         chunk_overlap_ms = getattr(args, "chunk_overlap_ms", 200)
#         self.segmenter = StreamingSegmenter(a_cfg, max_chunk_ms=max_chunk_ms, overlap_ms=chunk_overlap_ms)

#         self.transcriber = Transcriber(w_cfg)
#         self.translator = Translator(m_cfg)
#         self.glossary = Glossary(glossary_path, m_cfg.tgt_lang)

#         # TTS selection
#         if p_cfg.engine == "silero" and not p_cfg.enabled is False:
#             self.tts = SileroTTS(speaker=p_cfg.tts_voice, sample_rate=24000)
#             self.voice_cycle = ["aidar","eugene","nikolay"]
#         elif p_cfg.engine == "piper" and not p_cfg.enabled is False:
#             self.tts = PiperTTS(p_cfg)
#             self.voice_cycle = []
#         else:
#             class _NoTTS: 
#                 def speak(self, text: str): pass
#                 def toggle_mute(self): return False
#                 def set_voice(self, v): pass
#             self.tts = _NoTTS()
#             self.voice_cycle = []

#         # Queues / threads
#         self.q = queue.Queue()
#         self.translate_q: queue.Queue = queue.Queue(maxsize=6)
#         self.print_q: queue.Queue = queue.Queue(maxsize=12)
#         self.tts_q: queue.Queue = queue.Queue(maxsize=2)  # keep last 2, drop oldest if full
#         self.stop_event = threading.Event()

#         # Modes / state
#         self.streaming_mode = "fast" if getattr(args, "streaming_mode", "fast") == "fast" else "natural"
#         self._last_src = ("", 0.0)
#         self._hold_buffer: Optional[Tuple[str,str,float,float,float,float]] = None  # (src,tgt,asr,wait,mt,total)

#         # Warm-ups
#         _ = self.translator.translate("Hello")
#         dummy = np.zeros(int(self.audio_cfg.samplerate * 0.5), dtype=np.float32)
#         _ = self.transcriber.transcribe(dummy, self.audio_cfg.samplerate)

#         # Workers
#         threading.Thread(target=self._mt_worker, daemon=True).start()
#         threading.Thread(target=self._tts_worker, daemon=True).start()
#         if _KEYBOARD_OK:
#             threading.Thread(target=self._hotkeys_worker, daemon=True).start()
#         else:
#             threading.Thread(target=self._console_keys_worker, daemon=True).start()

#     # ---- workers ----
#     def _mt_worker(self):
#         while not self.stop_event.is_set():
#             try:
#                 src_text, asr_rt, start, mid = self.translate_q.get(timeout=0.1)
#             except queue.Empty:
#                 continue
#             t0 = time.time()
#             tgt_text = self.translator.translate(src_text)
#             tgt_text = self.glossary.apply(tgt_text)
#             t1 = time.time()
#             item = (src_text, tgt_text, asr_rt, mid - start - asr_rt, t1 - mid, t1 - start)
#             try:
#                 self.print_q.put_nowait(item)
#             except queue.Full:
#                 pass

#     def _tts_worker(self):
#         while not self.stop_event.is_set():
#             try:
#                 text = self.tts_q.get(timeout=0.1)
#             except queue.Empty:
#                 continue
#             try:
#                 self.tts.speak(text)
#             except Exception as e:
#                 console.print(f"[yellow]TTS error: {e}[/yellow]")

#     def _hotkeys_worker(self):
#         console.print("[magenta]Hotkeys:[/] F1 mute/unmute • F2 change voice • F3 fast/natural")
#         while not self.stop_event.is_set():
#             try:
#                 if keyboard.is_pressed("F1"):
#                     muted = self.tts.toggle_mute()
#                     console.print(f"[cyan]TTS {'muted' if muted else 'unmuted'}[/cyan]")
#                     time.sleep(0.3)
#                 if keyboard.is_pressed("F2") and self.voice_cycle:
#                     cur = getattr(self.tts, "speaker", self.voice_cycle[0])
#                     idx = (self.voice_cycle.index(cur) + 1) % len(self.voice_cycle) if cur in self.voice_cycle else 0
#                     self.tts.set_voice(self.voice_cycle[idx])
#                     console.print(f"[cyan]Voice → {self.voice_cycle[idx]}[/cyan]")
#                     time.sleep(0.3)
#                 if keyboard.is_pressed("F3"):
#                     self.streaming_mode = "natural" if self.streaming_mode == "fast" else "fast"
#                     console.print(f"[cyan]Streaming mode → {self.streaming_mode}[/cyan]")
#                     time.sleep(0.3)
#             except Exception:
#                 time.sleep(0.1)

#     def _console_keys_worker(self):
#         # Fallback if 'keyboard' isn't available: poll stdin without blocking TTS/ASR
#         console.print("[magenta]Hotkeys (console):[/] type f1/f2/f3 + Enter (focus this window)")
#         import sys, select
#         while not self.stop_event.is_set():
#             try:
#                 if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
#                     cmd = sys.stdin.readline().strip().lower()
#                     if cmd == "f1":
#                         muted = self.tts.toggle_mute(); console.print(f"[cyan]TTS {'muted' if muted else 'unmuted'}[/cyan]")
#                     elif cmd == "f2" and self.voice_cycle:
#                         cur = getattr(self.tts, "speaker", self.voice_cycle[0])
#                         idx = (self.voice_cycle.index(cur) + 1) % len(self.voice_cycle) if cur in self.voice_cycle else 0
#                         self.tts.set_voice(self.voice_cycle[idx]); console.print(f"[cyan]Voice → {self.voice_cycle[idx]}[/cyan]")
#                     elif cmd == "f3":
#                         self.streaming_mode = "natural" if self.streaming_mode == "fast" else "fast"
#                         console.print(f"[cyan]Streaming mode → {self.streaming_mode}[/cyan]")
#             except Exception:
#                 time.sleep(0.1)

#     # ---- audio I/O ----
#     @staticmethod
#     def _is_silent(x: np.ndarray, thresh_db: float = -45.0) -> bool:
#         rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
#         db = 20.0 * np.log10(rms)
#         return db < thresh_db

#     def audio_callback(self, indata, frames, time_info, status):
#         if status:
#             console.log(f"[yellow]Audio status: {status}[/]")
#         audio = np.clip(indata[:, 0], -1.0, 1.0)
#         pcm16 = (audio * 32768.0).astype(np.int16).tobytes()
#         seg = self.segmenter.process_block(pcm16)
#         if seg is not None:
#             self.q.put(seg)

#     # ---- main loop ----
#     def run(self):
#         ac = self.audio_cfg
#         sd.default.samplerate = ac.samplerate
#         sd.default.channels = ac.channels
#         sd.default.latency = 'low'
#         if ac.device is not None:
#             sd.default.device = ac.device

#         console.print(Panel.fit("[bold green]Starting live translation[/bold green]\nPress [b]Ctrl+C[/b] to stop.", box=box.ROUNDED))

#         with sd.InputStream(callback=self.audio_callback, dtype="float32", blocksize=int(ac.samplerate * ac.block_ms / 1000)):
#             try:
#                 while not self.stop_event.is_set():
#                     # Drain completed translations and decide when to speak
#                     try:
#                         while True:
#                             src, tgt, asr_rt, wait_gap, mt_rt, total = self.print_q.get_nowait()
#                             self._print_result(src, tgt, asr_rt, wait_gap, mt_rt, total)

#                             speak_now = True
#                             if self.streaming_mode == "natural":
#                                 # Hold if not final and pause is very short
#                                 if not _is_final(src) and wait_gap < 0.18:
#                                     # buffer one chunk (keep latest only)
#                                     self._hold_buffer = (src, tgt, asr_rt, wait_gap, mt_rt, total)
#                                     speak_now = False
#                                 else:
#                                     # if we had a hold and now we see a more "final" chunk, flush the held first
#                                     if self._hold_buffer:
#                                         try:
#                                             self.tts_q.put_nowait(self._hold_buffer[1])
#                                         except queue.Full:
#                                             pass
#                                         self._hold_buffer = None

#                             if speak_now:
#                                 # enqueue TTS (drop oldest if queue full)
#                                 try:
#                                     self.tts_q.put_nowait(tgt)
#                                 except queue.Full:
#                                     try:
#                                         _ = self.tts_q.get_nowait()
#                                         self.tts_q.put_nowait(tgt)
#                                     except Exception:
#                                         pass
#                     except queue.Empty:
#                         pass

#                     # Get next audio segment
#                     try:
#                         segment = self.q.get(timeout=0.05)
#                     except queue.Empty:
#                         continue

#                     if LiveTranslatorApp._is_silent(segment, -45.0):
#                         continue

#                     start = time.time()
#                     src_text, asr_rt = self.transcriber.transcribe(segment, self.audio_cfg.samplerate)
#                     mid = time.time()

#                     # Guards: skip junk, duplicates, non-English
#                     if not src_text or len(src_text.strip()) < 3 or len(src_text.split()) < 2:
#                         continue
#                     if not mostly_latin(src_text):
#                         continue
#                     core = src_text.strip().strip(".?!,:;—-")
#                     if not core:
#                         continue
#                     last_text, last_t = self._last_src
#                     now = time.time()
#                     norm = core.lower()
#                     if norm == last_text and (now - last_t) < 5.0:
#                         continue
#                     self._last_src = (norm, now)

#                     src_text = fix_refs(src_text)

#                     # Send to MT worker
#                     try:
#                         self.translate_q.put_nowait((src_text, asr_rt, start, mid))
#                     except queue.Full:
#                         pass

#             except KeyboardInterrupt:
#                 console.print("\n[cyan]Stopping...[/cyan]")
#                 self.stop_event.set()

#     def _print_result(self, src: str, tgt: str, asr_rt: float, wait_gap: float, mt_rt: float, total: float):
#         if not src.strip():
#             return
#         tbl = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0,1))
#         tbl.add_row("[bold]Source[/bold]", src.strip())
#         tbl.add_row("[bold]Translation[/bold]", f"[green]{tgt.strip()}[/green]")
#         timing = f"ASR {asr_rt*1000:.0f} ms • MT {mt_rt*1000:.0f} ms • Total {total*1000:.0f} ms"
#         console.print(Panel(tbl, title=timing, border_style="blue"))

# # ---------- CLI ----------
# def parse_args():
#     p = argparse.ArgumentParser(description="Live offline EN→RU speech translation (streaming + hotkeys)")
#     # Streaming & timing
#     p.add_argument("--streaming-mode", choices=["fast","natural"], default="fast", help="Output style")
#     p.add_argument("--max-chunk-ms", type=int, default=900, help="Max duration per streaming chunk")
#     p.add_argument("--chunk-overlap-ms", type=int, default=200, help="Overlap kept between chunks")
#     # Core pipeline
#     p.add_argument("--asr-lang", default="en", help="Force ASR language ('en' or 'None' to autodetect)")
#     p.add_argument("--whisper", default="small", help="Whisper size: tiny/base/small/medium/large-v3")
#     p.add_argument("--device", default="auto", help="ASR device: auto|cuda|cpu")
#     p.add_argument("--beam", type=int, default=1, help="ASR beam size (1 = greedy fastest)")
#     p.add_argument("--vad", action="store_true", help="Also enable Whisper's internal VAD")
#     p.add_argument("--block-ms", type=int, default=10, help="Audio block size for VAD (10/20/30 ms)")
#     p.add_argument("--silence-ms", type=int, default=180, help="Silence to end segment")
#     p.add_argument("--vad-aggr", type=int, default=3, help="WebRTC VAD aggressiveness 0-3")
#     # TTS / MT / Glossary
#     p.add_argument("--tts-engine", choices=["silero", "piper"], default="silero", help="Choose TTS engine")
#     p.add_argument("--tts-voice", default="aidar", help="TTS voice (silero: aidar/eugene/nikolay/baya/xenia)")
#     p.add_argument("--glossary", type=str, default="glossary.ru.json", help="Glossary JSON path")
#     p.add_argument("--mt-model", type=str, default="Helsinki-NLP/opus-mt-en-ru", help="MT model name")
#     # Piper paths (ignored for silero)
#     p.add_argument("--no-tts", action="store_true", help="Disable TTS altogether")
#     p.add_argument("--piper-exe", type=str, default=os.getenv("PIPER_EXE"))
#     p.add_argument("--piper-voice", type=str, default=os.getenv("VOICE_PATH"))
#     return p.parse_args()

# def main():
#     args = parse_args()

#     # Build configs
#     a_cfg = AudioConfig(block_ms=args.block_ms, max_silence_ms=args.silence_ms, vad_aggressiveness=args.vad_aggr)
#     w_cfg = WhisperConfig(model_size=args.whisper, device=args.device, beam_size=args.beam, vad_filter=args.vad,
#                           language=None if (args.asr_lang is None or str(args.asr_lang).lower()=="none") else args.asr_lang)
#     m_cfg = MTConfig(model_name=args.mt_model, src_lang=nllb_code("en"), tgt_lang=nllb_code("ru"))
#     p_cfg = PiperConfig(enabled=not args.no_tts, exe_path=args.piper_exe, voice_path=args.piper_voice,
#                         engine=args.tts_engine, tts_voice=args.tts_voice)

#     # Pass args into app class
#     LiveTranslatorApp.args = args
#     app = LiveTranslatorApp(a_cfg, w_cfg, m_cfg, p_cfg, Path(args.glossary))
#     app.run()

# if __name__ == "__main__":
#     main()



