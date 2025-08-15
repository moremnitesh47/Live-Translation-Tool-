
# Live Church Translation (EN→RU, offline)

Low-latency live translation for church services.  
**Pipeline:** Mic → VAD (WebRTC) → ASR (faster-whisper) → MT (Marian/NLLB) → TTS (Silero/Piper) → **LAN broadcast** to phones via a simple web page (no app).

> Built by **Nitesh Morem**. • Docs under CC BY 4.0 • Code under **MIT License**.

---

## ✨ Features
- Runs offline after first model download (no cloud calls)
- Fast Whisper ASR + WebRTC VAD
- English → Russian out of the box (swap models for other pairs)
- CPU-friendly **Silero TTS** (optional **Piper**)
- Phone listeners open `http://<server-ip>:8765` and tap **Join the Russian Channel**

> **Important:** Any IP shown in demos/printouts (e.g., `http://192.168.0.10:8765`) is an **example**.  
> The **church will provide the actual IP and port** on the day.
> As this tool was developed for use case of an adventist church in Cyprus to help translate the devotion or sermon for large number of people who dont speak English especially the old citizens.

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
````

### Open on phones (same Wi-Fi)

`http://<server-ip>:8765` → tap **Join the Russian Channel**.

---

## ⚙️ Common options

```bash
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
--glossary glossary.ru.json    # path to your glossary file
```

**Piper (optional):**

```bash
--piper-exe /path/to/piper --piper-voice /path/to/voice.onnx
# or set env:
PIPER_EXE=...  VOICE_PATH=...
```

---

## 📁 Glossary (term fixes after MT)

Create a JSON file and pass it via `--glossary glossary.ru.json`:

```json
{
  "target_lang": "rus_Cyrl",
  "replacements": {
    "Holy Spirit": "Святой Дух",
    "Gospel": "Евангелие"
  }
}
```

---

## 🖨 Printables for members (EN & RU)

One-page sheets with **4 steps + Quick Help**.
They clearly state the printed IP is **example only**; the church provides the actual address.

Suggested files:

* `printables/Live-Translation-QuickStart-EN.html`
* `printables/Live-Translation-QuickStart-RU.html`

Print at 100% scale (portrait). You can **type** your real Wi-Fi name, password, and server address in the files before printing, or write them by hand.

---

## 🔊 Hotkeys (if `keyboard` is installed)

* **F1** – mute/unmute broadcast
* **F2** – switch voice (male ↔ female)
* **ESC twice** – quit

---

## 🛠 Troubleshooting

* **No sound on phone:** same Wi-Fi, reload the page, tap **Join** again; check mute switch & volume.
* **Choppy audio:** move closer to router, turn off VPN/hotspot, pause downloads.
* **GPU not used:** try `--device cuda`, verify drivers/CUDA; otherwise use CPU.
* **Wrong mic/device:** set OS input device and/or `sounddevice` device index.

---

## 🔐 Privacy

The web page is **receive-only**; it **does not** record listener microphones.

---

## 🧾 License

* **Code:** MIT (see `LICENSE`)

---

## 🙌 Attribution (optional)

Built by **Nitesh Morem**.
AI-generated output was reviewed and validated by the author.

---
