#!/usr/bin/env python3
"""
Environment checker for Live Church Translation

What it does:
- Prints Python & OS info
- Shows versions of key packages used by live_translate.py
- Reports GPU/CUDA status (torch + ctranslate2)
- Summarizes audio input devices (sounddevice)
- Shows relevant env vars
- Sanity-checks WebRTC VAD frame sizes (10/20/30 ms @ 16kHz)
- (Optional) writes a pinned requirements file for just this project’s deps
- (Optional) writes a JSON report

Usage:
  python env_check.py
  python env_check.py --write-reqs requirements_project.txt
  python env_check.py --json env_report.json
"""

import sys, os, platform, json, argparse
from typing import Optional, Dict, Any

# -------- helpers
def v(name: str) -> Optional[str]:
    """Return distribution version if installed, else None."""
    try:
        from importlib.metadata import version, PackageNotFoundError
    except Exception:
        try:
            from importlib_metadata import version, PackageNotFoundError  # backport
        except Exception:
            return None
    try:
        return version(name)
    except Exception:
        return None

def safe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

def yesno(b: Optional[bool]) -> str:
    return "yes" if b else "no"

def try_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

# -------- collect info
def collect_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    # Python & OS
    info["python"] = {
        "version": sys.version.split()[0],
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "arch": platform.machine(),
    }
    info["os"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
    }

    # Packages (versions)
    packages = [
        "numpy", "rich", "sounddevice", "webrtcvad", "aiohttp", "soundfile",
        "faster-whisper", "ctranslate2", "transformers", "keyboard", "torch",
    ]
    info["packages"] = {p: v(p) for p in packages}

    # Torch / CUDA
    tor = safe_import("torch")
    cuda = {
        "installed": tor is not None,
        "version": getattr(tor, "__version__", None) if tor else None,
        "cuda_available": bool(getattr(tor.cuda, "is_available", lambda: False)()) if tor else False,
        "cuda_version": getattr(getattr(tor, "version", None), "cuda", None) if tor else None,
        "gpu_count": getattr(tor.cuda, "device_count", lambda: 0)() if tor else 0,
        "gpus": [],
    }
    if tor and cuda["cuda_available"]:
        try:
            cuda["gpus"] = [tor.cuda.get_device_name(i) for i in range(cuda["gpu_count"])]
        except Exception:
            pass
    info["torch_cuda"] = cuda

    # ctranslate2 (used by faster-whisper)
    ct2 = safe_import("ctranslate2")
    ct2_info = {
        "installed": ct2 is not None,
        "version": v("ctranslate2"),
        "cuda_device_count": 0,
    }
    if ct2:
        try:
            ct2_info["cuda_device_count"] = int(getattr(ct2, "get_cuda_device_count", lambda: 0)())
        except Exception:
            pass
    info["ctranslate2"] = ct2_info

    # Audio devices
    sd = safe_import("sounddevice")
    audio = {"default_input": None, "input_count": 0, "inputs": []}
    if sd:
        try:
            devs = sd.query_devices()
            audio["input_count"] = sum(1 for d in devs if d.get("max_input_channels", 0) > 0)
            # default device index may be tuple (input, output)
            try:
                defaults = sd.default.device
                default_in = defaults[0] if isinstance(defaults, (list, tuple)) else defaults
            except Exception:
                default_in = None
            audio["default_input"] = default_in
            # list up to 5 input devices
            for i, d in enumerate(devs):
                if d.get("max_input_channels", 0) > 0:
                    audio["inputs"].append({
                        "index": i,
                        "name": d.get("name"),
                        "max_input_channels": d.get("max_input_channels"),
                        "default_samplerate": d.get("default_samplerate"),
                    })
                    if len(audio["inputs"]) >= 5:
                        break
        except Exception as e:
            audio["error"] = str(e)
    else:
        audio["error"] = "sounddevice not installed"
    info["audio"] = audio

    # WebRTC VAD sanity (frame sizes)
    vad_ok = {"10ms": None, "20ms": None, "30ms": None, "error": None}
    webrtcvad = safe_import("webrtcvad")
    if webrtcvad:
        try:
            vad = webrtcvad.Vad(2)
            sr = 16000
            for ms in (10, 20, 30):
                n = int(sr * ms / 1000)
                frame = (b"\x00\x00") * n  # PCM16 zeros
                try:
                    # Just call is_speech to ensure no errors are raised.
                    vad.is_speech(frame, sr)
                    vad_ok[f"{ms}ms"] = True
                except Exception:
                    vad_ok[f"{ms}ms"] = False
        except Exception as e:
            vad_ok["error"] = str(e)
    else:
        vad_ok["error"] = "webrtcvad not installed"
    info["webrtc_vad"] = vad_ok

    # soundfile / libsndfile
    sf = safe_import("soundfile")
    sf_info = {"version": v("soundfile")}
    if sf:
        sf_info["libsndfile_version"] = getattr(sf, "__libsndfile_version__", None) or getattr(sf, "libsndfile_version", None)
    info["soundfile_info"] = sf_info

    # Env vars relevant to the app
    env_keys = [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM",
        "HF_HUB_DISABLE_SYMLINKS_WARNING", "HF_HOME", "TORCH_HOME",
        "CUDA_VISIBLE_DEVICES", "PIPER_EXE", "VOICE_PATH",
    ]
    info["env"] = {k: os.environ.get(k) for k in env_keys}

    return info

# -------- requirements writer
PROJECT_PKGS = [
    "numpy",
    "rich",
    "sounddevice",
    "webrtcvad",
    "aiohttp",
    "soundfile",
    "faster-whisper",
    "ctranslate2",
    "transformers",
    "keyboard",   # optional; remove if you don't want hotkeys
    "torch",      # include pin, but install channel (cpu/cuXXX) is up to you
]

def write_requirements(path: str, info: Dict[str, Any]) -> None:
    lines = []
    pkgs = info.get("packages", {})
    for name in PROJECT_PKGS:
        ver = pkgs.get(name)
        if ver:
            lines.append(f"{name}=={ver}")
        else:
            lines.append(f"# {name}  # not installed")
    # add tip comments for torch channel
    lines.append("")
    lines.append("# NOTE: For torch, you may prefer:")
    lines.append("#   CPU:  pip install --index-url https://download.pytorch.org/whl/cpu torch")
    lines.append("#   CUDA: pip install --index-url https://download.pytorch.org/whl/cu121 torch")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# -------- main
def main():
    ap = argparse.ArgumentParser(description="Check environment for Live Church Translation")
    ap.add_argument("--write-reqs", metavar="FILE", help="Write a pinned requirements file for this project")
    ap.add_argument("--json", metavar="FILE", help="Write a JSON report to FILE")
    args = ap.parse_args()

    info = collect_info()

    # Pretty print
    print("\n=== Python & OS ===")
    print(f"Python: {info['python']['version']} ({info['python']['implementation']}, {info['python']['arch']})")
    print(f"Executable: {info['python']['executable']}")
    print(f"OS: {info['os']['system']} {info['os']['release']}")

    print("\n=== Packages ===")
    for k, ver in info["packages"].items():
        print(f"{k:16s} {ver or 'not installed'}")

    print("\n=== Torch / CUDA ===")
    t = info["torch_cuda"]
    print(f"torch installed:  {yesno(t['installed'])}  version: {t.get('version')}")
    print(f"CUDA available:   {yesno(t['cuda_available'])}  cuda_version: {t.get('cuda_version')}")
    print(f"GPU count:        {t.get('gpu_count',0)}  names: {t.get('gpus')}")

    print("\n=== ctranslate2 ===")
    c = info["ctranslate2"]
    print(f"installed: {yesno(c['installed'])}  version: {c.get('version')}  cuda_devices: {c.get('cuda_device_count')}")

    print("\n=== Audio (sounddevice) ===")
    a = info["audio"]
    if "error" in a and a["error"]:
        print("error:", a["error"])
    else:
        print(f"default input index: {a['default_input']}")
        print(f"input device count: {a['input_count']}")
        for d in a["inputs"]:
            print(f"  [{d['index']}] {d['name']}  (max_in={d['max_input_channels']}, default_sr={d['default_samplerate']})")

    print("\n=== WebRTC VAD sanity (16kHz zeros) ===")
    vad = info["webrtc_vad"]
    if vad.get("error"):
        print("error:", vad["error"])
    else:
        print(f"10ms: {yesno(vad['10ms'])}  20ms: {yesno(vad['20ms'])}  30ms: {yesno(vad['30ms'])}")

    print("\n=== soundfile / libsndfile ===")
    sfi = info["soundfile_info"]
    print(f"soundfile: {sfi.get('version')}  libsndfile: {sfi.get('libsndfile_version')}")

    print("\n=== Relevant environment variables ===")
    for k, val in info["env"].items():
        print(f"{k:28s} {val}")

    if args.write_reqs:
        write_requirements(args.write_reqs, info)
        print(f"\nWrote pinned requirements to: {args.write_reqs}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"Wrote JSON report to: {args.json}")

if __name__ == "__main__":
    main()
