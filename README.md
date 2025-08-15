# Live Church Translation (EN→RU, offline)

Low-latency live translation for church services.  
**Pipeline:** Mic → VAD (WebRTC) → ASR (faster-whisper) → MT (Marian/NLLB) → TTS (Silero/Piper) → **LAN broadcast** to phones via a simple web page (no app).

> Built by **Nitesh Morem**. Docs under **CC BY 4.0**; code under **MIT**.

---

## ✨ Features
- Runs offline after first model download (no cloud calls)
- Fast Whisper ASR with WebRTC VAD
- English → Russian out of the box (swap models for other pairs)
- CPU-friendly **Silero TTS** (optional **Piper**)
- Phone listeners open `http://<server-ip>:8765` and tap **Join the Russian Channel**

> **Important:** Any IP shown in demos/printouts (e.g., `http://192.168.0.10:8765`) is an **example**.  
> The **church will provide the actual IP and port** to use on the day.

---

## 📦 Requirements
- Python **3.9+** (Windows/macOS/Linux)
- Optional NVIDIA GPU (CUDA) for faster Whisper
- `pip install -r requirements.txt`

---

## 🚀 Quick start
```bash
# 1) (optional) create a virtual env
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) install dependencies
pip install -r requirements.txt

# 3) run
python live_translate.py --whisper small --device auto --tts-engine silero --lan-port 8765
```

#On the same Wi-Fi, open on phones:
`http://<server-ip>:8765` → tap Join the Russian Channel.

##Common options
# Device
--device auto|cuda|cpu

# Whisper size (speed/quality)
--whisper tiny|base|small|medium|large-v3
--beam 1                       # 1 = greedy (fastest)

# VAD
--block-ms 10|20|30            # WebRTC VAD frame size
--vad-aggr 0..3                # aggressiveness (3 = most strict)

# TTS
--tts-engine silero|piper
--tts-voice aidar|eugene|nikolay|baya|xenia
--female-voice xenia           # quick toggle voice

# LAN
--lan-port 8765
--no-lan                       # disable web broadcast

# Glossary
--glossary glossary.ru.json
{
  "target_lang": "rus_Cyrl",
  "replacements": {
    "Holy Spirit": "Святой Дух",
    "Gospel": "Евангелие"
  }
}


##🔐 Privacy
- The web page is receive-only; it does not record listener microphones.

##🧾 License

- Code: MIT (see LICENSE)

