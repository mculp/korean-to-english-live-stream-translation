# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "streamlink>=8.5",
#   "av",
#   "numpy",
#   "sounddevice",
#   "silero-vad",
#   "edge-tts",
#   "mlx-whisper; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "faster-whisper; sys_platform != 'darwin' or platform_machine != 'arm64'",
# ]
# ///
"""kodub: live Korean or Mandarin -> English voice dub for streamlink streams.

Korean is the default. Mandarin Warcraft 3 (Douyu, Huya, Douyin) uses --lang zh.
Streamlink 8.5+ has those plugins; the opener sniffs FLV vs MPEG-TS.

Only speech recognised as the chosen source language is dubbed. English speech and
non-speech game audio are skipped. For the Warcraft 3 glossary and --context / --overdub,
use kodub_llm.py.
"""
import argparse, asyncio, io, platform, queue, re, sys, threading, time

import av
import numpy as np

SR = 16000
HANGUL = re.compile(r"[\uac00-\ud7a3]")
HAN = re.compile(r"[\u4e00-\u9fff]")
# Common Whisper hallucinations on music/silence
JUNK = ["시청해주셔서", "구독과 좋아요", "MBC 뉴스", "자막", "감사합니다.", "다음 영상에서"]
ZH_JUNK_RES = [
    re.compile(r"字幕由.{0,40}提供"),
    re.compile(r"请不吝点赞"),
    re.compile(r"订阅[,，、\s]*转发"),
    re.compile(r"转发[,，、\s]*打赏"),
    re.compile(r"明镜与点点"),
    re.compile(r"点点栏目"),
    re.compile(r"感谢您的?(观看|收看)"),
    re.compile(r"谢谢观看"),
    re.compile(r"感谢收看"),
    re.compile(r"下期再见"),
    re.compile(r"本期视频"),
    re.compile(r"字幕组"),
    re.compile(r"中文字幕"),
    re.compile(r"优优独播"),
    re.compile(r"Amara\.org", re.I),
    re.compile(r"我的频道"),
    re.compile(r"欢迎订阅"),
    re.compile(r"打赏支持"),
    re.compile(r"支持明镜"),
]
ZH_ASR_PROMPT = (
    "魔兽争霸3解说. Infi, Fly100%, TH000, Lyn, Moon, 120, LawLiet, Colorful, YouTiao, 油条, eer0, LioN, Happy, Fortitude, Sok. "
    "人族, 兽族, 亡灵, 暗夜, 大法师, 山丘, 剑圣, 死骑, 恶魔猎手, 小鹿, 月女, 遗忘骑, 分矿, 二本, 三本, 偷家."
)
LANGS = {
    "ko": {"whisper": "ko", "script": HANGUL, "google": "ko", "mymemory": "ko|en", "label": "KO"},
    "zh": {"whisper": "zh", "script": HAN, "google": "zh-CN", "mymemory": "zh-CN|en", "label": "ZH"},
}


def norm_lang(lang):
    if not lang:
        return "?"
    lang = str(lang).lower().replace("_", "-")
    if lang in ("zh", "zh-cn", "zh-hans", "chinese", "cmn", "mandarin"):
        return "zh"
    if lang in ("ko", "korean"):
        return "ko"
    return lang.split("-")[0]


def is_junk(text, lang):
    if lang == "ko":
        return any(j in (text or "") for j in JUNK)
    return any(r.search(text or "") for r in ZH_JUNK_RES)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- audio input
class _FD:
    def __init__(self, fd): self.fd = fd
    def read(self, n=-1): return self.fd.read(n if n and n > 0 else 65536)
    def readable(self): return True


class _Prefixed:
    def __init__(self, head, fd):
        self._head = head or b""
        self._fd = fd

    def read(self, n=-1):
        if n is None or n < 0:
            n = 65536
        out = b""
        if self._head:
            take = self._head[:n]
            self._head = self._head[n:]
            out = take
            if len(out) >= n:
                return out
            n -= len(out)
        chunk = self._fd.read(n)
        return out + (chunk or b"")

    def readable(self):
        return True


def sniff_container(fd):
    """Douyu, Huya and Douyin are FLV. SOOP/Afreeca via streamlink is MPEG-TS."""
    head = b""
    while len(head) < 4:
        b = fd.read(4 - len(head))
        if not b:
            break
        head += b
    if head.startswith(b"FLV"):
        fmt = "flv"
    elif head.startswith(b"OggS"):
        fmt = "ogg"
    elif head.startswith(b"RIFF"):
        fmt = "wav"
    elif head[:1] == b"\x47":
        fmt = "mpegts"
    else:
        fmt = "mpegts"
        if head:
            log(f"unrecognized stream header {head[:8]!r}; trying mpegts")
    return fmt, _Prefixed(head, fd)


def audio_source(url, quality, out_q, stop):
    """Pull the live stream with streamlink, decode audio to 16 kHz mono float32 and push into out_q."""
    import streamlink
    while not stop.is_set():
        try:
            session = streamlink.Streamlink()
            session.set_option("http-timeout", 20)
            streams = session.streams(url)
            if not streams:
                log("stream offline / no streams, retrying in 15s"); time.sleep(15); continue
            picked = quality if quality in streams else ("worst" if "worst" in streams else next(iter(streams)))
            raw = streams[picked].open()
            fmt, replay = sniff_container(_FD(raw))
            log(f"connected ({picked}, {fmt})")
            kwargs = {"format": fmt}
            if fmt == "flv":
                kwargs["options"] = {"fflags": "nobuffer", "flags": "low_delay"}
            c = av.open(replay, **kwargs)
            rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
            for frame in c.decode(audio=0):
                if stop.is_set(): break
                for f in rs.resample(frame):
                    out_q.put(f.to_ndarray().reshape(-1).astype(np.float32))
        except Exception as e:
            log("input error:", repr(e), "- reconnecting in 3s"); time.sleep(3)


# ---------------------------------------------------------------- VAD segmenter
def segmenter(in_q, seg_q, stop, max_len=12.0, min_len=0.6):
    from silero_vad import load_silero_vad, VADIterator
    import torch
    model = load_silero_vad()
    vad = VADIterator(model, threshold=0.5, sampling_rate=SR, min_silence_duration_ms=450, speech_pad_ms=120)
    buf = np.zeros(0, np.float32)
    cur = None  # list of chunks while in speech
    pre = []    # small pre-roll
    while not stop.is_set():
        try: buf = np.concatenate([buf, in_q.get(timeout=1)])
        except queue.Empty: continue
        while len(buf) >= 512:
            ch, buf = buf[:512], buf[512:]
            ev = vad(torch.from_numpy(ch), return_seconds=False)
            if ev and "start" in ev and cur is None:
                cur = pre[-4:] + [ch]
            elif cur is not None:
                cur.append(ch)
                if (ev and "end" in ev) or len(cur) * 512 / SR >= max_len:
                    seg = np.concatenate(cur)
                    if len(seg) / SR >= min_len: seg_q.put((time.time(), seg))
                    cur = None
                    if not (ev and "end" in ev): vad.reset_states()
            pre.append(ch); pre = pre[-4:]


# ---------------------------------------------------------------- ASR
class ASR:
    def __init__(self, model, lang="ko"):
        self.mlx = platform.system() == "Darwin" and platform.machine() == "arm64"
        self.lang = lang if lang in LANGS else "ko"
        self.code = LANGS[self.lang]["whisper"]
        if self.mlx:
            import mlx_whisper
            self.m = mlx_whisper
            self.name = model or "mlx-community/whisper-large-v3-turbo"
            # warm-up / download
            self.m.transcribe(np.zeros(SR, np.float32), path_or_hf_repo=self.name, language=self.code)
        else:
            from faster_whisper import WhisperModel
            self.name = model or "small"
            self.m = WhisperModel(self.name, device="cpu", compute_type="int8")

    def _prompt(self):
        # Korean kodub.py never biased Whisper; only Mandarin gets a vocabulary prompt.
        return ZH_ASR_PROMPT if self.lang == "zh" else None

    def transcribe(self, audio):
        """returns (lang, text, no_speech_prob)"""
        ip = self._prompt()
        if self.mlx:
            kw = dict(path_or_hf_repo=self.name, condition_on_previous_text=False, temperature=0.0, verbose=None)
            if ip:
                kw["initial_prompt"] = ip
            r = self.m.transcribe(audio, **kw)
            segs = r.get("segments", [])
            nsp = min([s.get("no_speech_prob", 0) for s in segs] or [1])
            return norm_lang(r.get("language")), r.get("text", "").strip(), nsp
        kw = dict(beam_size=1, condition_on_previous_text=False, vad_filter=False)
        if ip:
            kw["initial_prompt"] = ip
        segs, info = self.m.transcribe(audio, **kw)
        segs = list(segs)
        nsp = min([s.no_speech_prob for s in segs] or [1])
        lang = norm_lang(info.language if info.language_probability >= 0.5 else "?")
        return lang, "".join(s.text for s in segs).strip(), nsp

    def translate(self, audio):
        if self.mlx:
            return self.m.transcribe(audio, path_or_hf_repo=self.name, language=self.code, task="translate",
                                     temperature=0.0, verbose=None)["text"].strip()
        segs, _ = self.m.transcribe(audio, language=self.code, task="translate", beam_size=1)
        return "".join(s.text for s in segs).strip()


# ---------------------------------------------------------------- translation
def google_translate(text, sl="ko"):
    import json, urllib.parse, urllib.request
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=" + sl + "&tl=en&dt=t&q=" + urllib.parse.quote(text)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.load(urllib.request.urlopen(req, timeout=5))
    return "".join(x[0] for x in data[0] if x and x[0]).strip()


def mymemory_translate(text, pair="ko|en"):
    import json, urllib.parse, urllib.request
    url = "https://api.mymemory.translated.net/get?langpair=" + pair + "&q=" + urllib.parse.quote(text)
    d = json.load(urllib.request.urlopen(url, timeout=5))
    return (d.get("responseData") or {}).get("translatedText", "").strip()


def translate_text(text, sl="ko", pair="ko|en", script=None):
    """Google (with one retry) -> MyMemory. Returns None if all fail."""
    script = HANGUL if script is None else script
    for fn, tries, arg in ((google_translate, 2, sl), (mymemory_translate, 1, pair)):
        for i in range(tries):
            try:
                en = fn(text, arg)
                if en and not script.search(en): return en
            except Exception as e:
                log(f"  ({fn.__name__} {type(e).__name__})")
            time.sleep(0.4)
    return None


# ---------------------------------------------------------------- TTS + playback
def eleven_pcm(text, voice_id, key):
    """ElevenLabs low-latency TTS (Flash v2.5), raw 24 kHz PCM."""
    import json, urllib.request
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?output_format=pcm_24000&optimize_streaming_latency=3"
    body = json.dumps({"text": text, "model_id": "eleven_flash_v2_5"}).encode()
    req = urllib.request.Request(url, data=body, headers={"xi-api-key": key, "Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=15).read()
    return np.frombuffer(raw[: len(raw) // 2 * 2], "<i2").astype(np.float32) / 32768


def say_pcm(text):
    """macOS built-in TTS fallback."""
    import os, subprocess, tempfile
    f = tempfile.mktemp(suffix=".aiff")
    subprocess.run(["say", "-o", f, text], check=True)
    try:
        c = av.open(f); rs = av.AudioResampler(format="flt", layout="mono", rate=24000)
        return np.concatenate([x.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for x in rs.resample(fr)])
    finally:
        os.unlink(f)


def tts_pcm(text, voice, rate):
    import edge_tts
    async def go():
        b = bytearray()
        async for ch in edge_tts.Communicate(text, voice, rate=rate).stream():
            if ch["type"] == "audio": b.extend(ch["data"])
        return bytes(b)
    mp3 = asyncio.run(go())
    c = av.open(io.BytesIO(mp3))
    rs = av.AudioResampler(format="flt", layout="mono", rate=24000)
    out = [f.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for f in rs.resample(fr)]
    return np.concatenate(out) if out else np.zeros(0, np.float32)


class Speaker:
    """One persistent output stream at the device's native rate with a generous buffer.

    The old code called sd.play(pcm, 24000) per line: a fresh tiny-buffer (~20 ms) stream at 24 kHz on a
    44.1/48 kHz (often AirPlay/Bluetooth) device, so CoreAudio resampled on the fly and the Python callback
    had to keep up while Whisper/VAD hogged the CPU/GPU -> crackles. Here we resample once with libswresample,
    keep one stream open, use large blocks + ~0.4 s latency, add short fades and a soft limiter."""

    def __init__(self, volume=1.0, device=None, latency=0.4, blocksize=2048):
        import sounddevice as sd
        self.sd, self.volume = sd, volume
        info = sd.query_devices(device, "output")
        self.sr = int(info["default_samplerate"]) or 48000
        self.buf = np.zeros(0, np.float32)
        self.lock = threading.Lock()
        self.idle = threading.Event(); self.idle.set()
        self.underflows = self.callbacks = self.late = 0
        self._last = None
        self.stream = sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32", device=device,
                                      blocksize=blocksize, latency=latency, callback=self._cb)
        self.stream.start()
        self.name = info["name"]
        self.slack = self.stream.latency + blocksize / self.sr  # max callback gap before audio runs dry

    def _cb(self, out, frames, t, status):
        self.callbacks += 1
        now = time.perf_counter()
        if status.output_underflow or (self._last and now - self._last > self.slack): self.underflows += 1
        self._last = now
        with self.lock:
            n = min(frames, len(self.buf))
            out[:n, 0] = self.buf[:n]
            out[n:, 0] = 0
            self.buf = self.buf[n:]
            if not len(self.buf): self.idle.set()

    def resample(self, pcm, rate=24000):
        if rate == self.sr or not len(pcm): return pcm.astype(np.float32)
        fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(pcm, np.float32).reshape(1, -1), format="flt", layout="mono")
        fr.sample_rate = rate
        rs = av.AudioResampler(format="flt", layout="mono", rate=self.sr)
        out = [f.to_ndarray().reshape(-1) for f in rs.resample(fr)] + [f.to_ndarray().reshape(-1) for f in rs.resample(None)]
        return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)

    def prepare(self, pcm, rate=24000):
        x = self.resample(np.nan_to_num(pcm), rate) * self.volume
        f = min(int(self.sr * 0.008), len(x) // 2)  # 8 ms fade in/out: no click at line edges
        if f > 0:
            ramp = np.linspace(0, 1, f, dtype=np.float32)
            x[:f] *= ramp; x[-f:] *= ramp[::-1]
        return np.tanh(x * 1.0).astype(np.float32) if np.abs(x).max() > 0.95 else x  # soft limit, never hard-clip

    def play(self, pcm, rate=24000):
        x = self.prepare(pcm, rate)
        with self.lock:
            self.buf = np.concatenate([self.buf, x, np.zeros(int(self.sr * 0.15), np.float32)])
            self.idle.clear()
        self.idle.wait()

    def close(self):
        self.idle.wait(5); time.sleep(0.5)
        self.stream.stop(); self.stream.close()


SPEAKER = None


def player(play_q, stop, mute, volume, save, device=None, latency=0.4):
    sp = None
    if not mute:
        global SPEAKER
        sp = SPEAKER = Speaker(volume, device, latency)
        log(f"audio out: {sp.name} @ {sp.sr} Hz, latency {sp.stream.latency:.2f}s")
    saved = []
    while not stop.is_set():
        try: pcm = play_q.get(timeout=0.5)
        except queue.Empty: continue
        if save is not None: saved.append(pcm); saved.append(np.zeros(6000, np.float32))
        if sp:
            before = sp.underflows
            sp.play(pcm)
            if sp.underflows > before: log(f"  (audio underflow x{sp.underflows - before})")
    if save is not None and saved:
        import wave
        a = (np.clip(np.concatenate(saved), -1, 1) * 32767).astype("<i2")
        with wave.open(save, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(a.tobytes())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Live Korean or Mandarin -> English voice dub for a stream URL")
    ap.add_argument("url")
    ap.add_argument("--lang", choices=["ko", "zh"], default="ko",
                    help="source language: ko (Korean, default) or zh (Mandarin). Glossary and --overdub are in kodub_llm.py")
    ap.add_argument("--quality", default="worst", help="stream quality to pull (audio is the same; default worst = least bandwidth)")
    ap.add_argument("--model", default=None, help="whisper model (default: large-v3-turbo via MLX on Apple Silicon, 'small' elsewhere)")
    ap.add_argument("--voice", default="en-US-GuyNeural", help="edge-tts voice, e.g. en-US-AriaNeural")
    ap.add_argument("--eleven-voice", default="JBFqnCBsd6RMkjVDRZzb",
                    help="ElevenLabs voice id (used only when ELEVENLABS_API_KEY is set)")
    ap.add_argument("--volume", type=float, default=1.0)
    ap.add_argument("--out-device", default=None, help="output device name/index (default: system output)")
    ap.add_argument("--out-latency", type=float, default=0.4, help="output buffer in seconds (raise if it crackles)")
    ap.add_argument("--translator", choices=["google", "whisper"], default="google",
                    help="google = transcribe then Google/MyMemory (better); whisper = whisper's built-in translate")
    ap.add_argument("--max-lag", type=float, default=12.0, help="drop segments older than this many seconds")
    ap.add_argument("--mute", action="store_true", help="don't play audio, just print (testing)")
    ap.add_argument("--save", default=None, help="also write dubbed audio to this .wav")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (testing)")
    a = ap.parse_args()

    prof = LANGS[a.lang]
    log("loading models (first run downloads ~1.5 GB)...")
    asr = ASR(a.model, lang=a.lang)
    log("ASR ready:", asr.name, "(MLX)" if asr.mlx else "(CPU)", "lang", a.lang)
    gt = a.translator == "google"
    import os
    ekey = os.environ.get("ELEVENLABS_API_KEY")
    log("TTS:", "ElevenLabs Flash v2.5" if ekey else "Edge neural voice " + a.voice)

    stop = threading.Event()
    in_q, seg_q, play_q = queue.Queue(), queue.Queue(), queue.Queue()
    for fn, args in [(audio_source, (a.url, a.quality, in_q, stop)), (segmenter, (in_q, seg_q, stop)),
                     (player, (play_q, stop, a.mute, a.volume, a.save, a.out_device, a.out_latency))]:
        threading.Thread(target=fn, args=args, daemon=True).start()

    t_end = time.time() + a.duration if a.duration else None
    lat = []
    try:
        while not (t_end and time.time() > t_end):
            try: t_seg, seg = seg_q.get(timeout=0.5)
            except queue.Empty: continue
            age = time.time() - t_seg
            if age > a.max_lag:
                log(f"[skip, {age:.0f}s behind]"); continue
            lang, text, nsp = asr.transcribe(seg)
            ok = norm_lang(lang) == prof["whisper"] and len(prof["script"].findall(text)) >= 2 and nsp < 0.6 \
                and not is_junk(text, a.lang)
            if not ok:
                if text: log(f"  - skip [{lang}] {text[:70]}")
                continue
            en = None
            if gt:
                en = translate_text(text, sl=prof["google"], pair=prof["mymemory"], script=prof["script"])
            if not en: en = asr.translate(seg)
            if not en or prof["script"].search(en) or not re.search(r"[A-Za-z]", en):
                log(f"  - translation failed for: {text[:60]}"); continue
            backlog = play_q.qsize() + seg_q.qsize()
            rate = f"+{min(10 + 15 * backlog, 50)}%"
            try:
                if ekey:
                    try: pcm = eleven_pcm(en, a.eleven_voice, ekey)
                    except Exception as e:
                        log("elevenlabs failed:", type(e).__name__); pcm = tts_pcm(en, a.voice, rate)
                else:
                    pcm = tts_pcm(en, a.voice, rate)
            except Exception as e:
                log("edge-tts failed:", type(e).__name__)
                if platform.system() != "Darwin": continue
                pcm = say_pcm(en)
            play_q.put(pcm)
            d = time.time() - t_seg
            lat.append(d)
            log(f"{prof['label']}: {text}\n         EN: {en}   (ready {d:.1f}s after speech ended)")
    except KeyboardInterrupt:
        pass
    stop.set()
    if lat: log(f"latency (speech end -> English audio ready): median {np.median(lat):.1f}s, max {max(lat):.1f}s, n={len(lat)}")
    if SPEAKER: log(f"audio: {SPEAKER.underflows} underflows/dropouts in {SPEAKER.callbacks} output callbacks")
    time.sleep(1.5)


if __name__ == "__main__":
    main()
