# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "streamlink>=7",
#   "av",
#   "numpy",
#   "sounddevice",
#   "silero-vad",
#   "edge-tts",
#   "mlx-whisper; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "faster-whisper; sys_platform != 'darwin' or platform_machine != 'arm64'",
# ]
# ///
"""kodub: live Korean -> English voice dub for SOOP/AfreecaTV (or any streamlink-supported) streams.

Only speech recognised as Korean is dubbed; English speech and non-speech game audio are skipped.
"""
import argparse, asyncio, io, platform, queue, re, sys, threading, time

import av
import numpy as np

SR = 16000
HANGUL = re.compile(r"[\uac00-\ud7a3]")
# Common Whisper hallucinations on music/silence
JUNK = ["시청해주셔서", "구독과 좋아요", "MBC 뉴스", "자막", "감사합니다.", "다음 영상에서"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- audio input
class _FD:
    def __init__(self, fd): self.fd = fd
    def read(self, n=-1): return self.fd.read(n if n and n > 0 else 65536)
    def readable(self): return True


def audio_source(url, quality, out_q, stop):
    """Pull the live stream with streamlink, decode audio to 16 kHz mono float32 and push into out_q."""
    import streamlink
    while not stop.is_set():
        try:
            streams = streamlink.Streamlink().streams(url)
            if not streams:
                log("stream offline / no streams, retrying in 15s"); time.sleep(15); continue
            st = streams.get(quality) or streams["worst"]
            fd = st.open()
            c = av.open(_FD(fd), format="mpegts")
            rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
            log(f"connected ({quality})")
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
ASR_PROMPT = "워크래프트3 중계. 소인, 영섭, 영삼, 032, TH, 파시오, 예진. 마킹, 코셀빵, 포세이큰, 투생템, 멀티, 런버밀, 헤드헌터, 프리스트, 그런트, 힐링웨이브."
_PROMPT_TERMS = [t.strip(" .") for t in re.split(r"[,.]", ASR_PROMPT) if t.strip(" .")]


def is_hallucination(text, prompt=ASR_PROMPT):
    """True if the transcript just echoes the prompt or is degenerate repetition."""
    t = text.strip()
    if not t: return True
    if "워크래프트3 중계" in t: return True
    # echo of the term list: 4+ prompt terms separated by commas
    hits = sum(1 for term in _PROMPT_TERMS if term in t)
    if hits >= 4 and t.count(",") >= 3: return True
    # repetition: a word/phrase making up most of the text
    words = t.split()
    if len(words) >= 6 and len(set(words)) / len(words) < 0.35: return True
    if re.search(r"(.{2,15}?)\1{3,}", t.replace(" ", "")): return True
    return False


class ASR:
    def __init__(self, model, prompt=True, ctx_lines=0):
        self.mlx = platform.system() == "Darwin" and platform.machine() == "arm64"
        self.prompt, self.ctx_lines, self.prev = prompt, ctx_lines, []
        if self.mlx:
            import mlx_whisper
            self.m = mlx_whisper
            self.name = model or "mlx-community/whisper-large-v3-mlx"
            self.m.transcribe(np.zeros(SR, np.float32), path_or_hf_repo=self.name, language="ko")  # warm-up / download
        else:
            from faster_whisper import WhisperModel
            self.name = model or "small"
            self.m = WhisperModel(self.name, device="cpu", compute_type="int8")

    def _initial_prompt(self):
        if not self.prompt: return None
        p = ASR_PROMPT
        if self.ctx_lines and self.prev: p += " " + " ".join(self.prev[-self.ctx_lines:])
        return p

    def transcribe(self, audio):
        """returns (lang, text, no_speech_prob)"""
        ip = self._initial_prompt()
        if self.mlx:
            r = self.m.transcribe(audio, path_or_hf_repo=self.name, condition_on_previous_text=False,
                                  temperature=0.0, verbose=None, initial_prompt=ip)
            segs = r.get("segments", [])
            nsp = min([s.get("no_speech_prob", 0) for s in segs] or [1])
            if any(s.get("compression_ratio", 0) > 2.4 for s in segs): nsp = 1
            lang, text = r.get("language"), r.get("text", "").strip()
        else:
            segs, info = self.m.transcribe(audio, beam_size=1, condition_on_previous_text=False, vad_filter=False,
                                           initial_prompt=ip)
            segs = list(segs)
            nsp = min([s.no_speech_prob for s in segs] or [1])
            lang = info.language if info.language_probability >= 0.5 else "?"
            text = "".join(s.text for s in segs).strip()
        if is_hallucination(text):
            return lang, "", 1.0
        if lang == "ko" and text: self.prev = (self.prev + [text])[-2:]
        return lang, text, nsp

    def translate(self, audio):
        if self.mlx:
            return self.m.transcribe(audio, path_or_hf_repo=self.name, language="ko", task="translate",
                                     temperature=0.0, verbose=None)["text"].strip()
        segs, _ = self.m.transcribe(audio, language="ko", task="translate", beam_size=1)
        return "".join(s.text for s in segs).strip()


# ---------------------------------------------------------------- translation
def google_translate(text):
    import json, urllib.parse, urllib.request
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=ko&tl=en&dt=t&q=" + urllib.parse.quote(text)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.load(urllib.request.urlopen(req, timeout=5))
    return "".join(x[0] for x in data[0] if x and x[0]).strip()


def mymemory_translate(text):
    import json, urllib.parse, urllib.request
    url = "https://api.mymemory.translated.net/get?langpair=ko|en&q=" + urllib.parse.quote(text)
    d = json.load(urllib.request.urlopen(url, timeout=5))
    return (d.get("responseData") or {}).get("translatedText", "").strip()


# ---------------------------------------------------------------- LLM translation (OpenRouter)
LLM_SYSTEM = """You translate live Korean esports/game-stream commentary (e.g. Warcraft 3 or StarCraft) into English, line by line, for text-to-speech.
The speaker is a Korean streamer/caster commentating on other players' games. Players are usually referred to by nickname (e.g. 소인, 영삼/영섭, 032, TH, 파시오).
Rules:
- Refer to players as he/they. Use "I" only when the caster clearly talks about himself (his own stream, plans, opinions).
- Keep player nicknames romanized and consistent with earlier lines; render game terms the way English players say them (unit/building names, "tier 2", "expansion", "creeping", "harass") and keep slang natural (e.g. 투생템 = going straight to tier 2 / skipping items, 멀티 = expansion, 견제 = harass).
- Warcraft 3 glossary: 포세이큰/포세이크 = Forsaken (a newly added race, so new Forsaken unit/hero names may appear; "he" when it means that player),
  마킹/MK = Mountain King, 코셀빵 (ASR variants: 코셀, 코셀팡, 코셀방) = Forsaken Paladin (a newly added hero), 휴먼 = Human, 오크 = Orc, 언데드 = Undead, 나엘 = Night Elf,
  팔라 = Paladin, 아크메이지/아크 = Archmage, 블마 = Blademaster, 데나 = Death Knight, 헤드헌터 = Headhunters, 그런트 = Grunts, 라이플맨 = Riflemen,
  프리스트 = Priests, 생추 = Sanctuary, 캐슬 = Castle, 공방 업 = attack/armor upgrades (1-1 = level 1 both), 렙/레벨 = hero level, 3,3렙 = both heroes level 3,
  땡X = going straight/mass X, 사냥 = creeping, 뒷방 = backdoor/base raid, 멀티 = expansion, 본진 = main base, 견제 = harass, 투생템 = tier-2 rush.
- Speech recognition may garble words; infer the most likely meaning from context rather than translating nonsense literally.
- Be concise and natural for speech. Output ONLY the English line, no quotes, notes or alternatives."""


def load_openrouter_key():
    import os
    k = os.environ.get("OPENROUTER_API_KEY")
    if k: return k
    for path in ("~/kodub/.env", "~/.env", "~/.hermes/.env"):
        try:
            for line in open(os.path.expanduser(path)):
                line = line.strip()
                if line.startswith(("OPENROUTER_API_KEY=", "export OPENROUTER_API_KEY=")):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return None


class LLMTranslator:
    def __init__(self, key, model, ctx=5):
        self.key, self.model, self.ctx = key, model, ctx
        self.history = []  # (ko, en)
        self.tokens_in = self.tokens_out = 0
        self.cost = 0.0

    def __call__(self, text):
        import json, urllib.request
        msgs = [{"role": "system", "content": LLM_SYSTEM}]
        for ko, en in self.history[-self.ctx:]:
            msgs += [{"role": "user", "content": ko}, {"role": "assistant", "content": en}]
        msgs.append({"role": "user", "content": text})
        body = json.dumps({"model": self.model, "messages": msgs, "max_tokens": 300, "temperature": 0.2,
                           "reasoning": {"enabled": False},
                           "provider": {"sort": "latency"},
                           "usage": {"include": True}}).encode()
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={
            "Authorization": "Bearer " + self.key, "Content-Type": "application/json",
            "X-Title": "kodub"})
        d = json.load(urllib.request.urlopen(req, timeout=6))
        en = (d["choices"][0]["message"].get("content") or "").strip().strip('"')
        u = d.get("usage") or {}
        self.tokens_in += u.get("prompt_tokens", 0); self.tokens_out += u.get("completion_tokens", 0)
        self.cost += float(u.get("cost") or 0)
        if en: self.history.append((text, en))
        return en


def translate_text(text):
    """Google (with one retry) -> MyMemory. Returns None if all fail."""
    for fn, tries in ((google_translate, 2), (mymemory_translate, 1)):
        for i in range(tries):
            try:
                en = fn(text)
                if en and not HANGUL.search(en): return en
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


def openrouter_tts_pcm(text, model, voice, key):
    """TTS via OpenRouter /audio/speech (e.g. elevenlabs/eleven-flash-v2.5). Returns 24 kHz mono float32."""
    import json, urllib.request
    body = json.dumps({"model": model, "input": text, "voice": voice, "response_format": "mp3"}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/audio/speech", data=body, headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json", "X-Title": "kodub"})
    mp3 = urllib.request.urlopen(req, timeout=10).read()
    c = av.open(io.BytesIO(mp3))
    rs = av.AudioResampler(format="flt", layout="mono", rate=24000)
    out = [f.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for f in rs.resample(fr)]
    if not out: raise RuntimeError("no audio")
    return np.concatenate(out)


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
    ap = argparse.ArgumentParser(description="Live Korean->English voice dub for a stream URL")
    ap.add_argument("url")
    ap.add_argument("--quality", default="worst", help="stream quality to pull (audio is the same; default worst = least bandwidth)")
    ap.add_argument("--asr-model", "--model", dest="model", default=None,
                    help="whisper model (default: mlx-community/whisper-large-v3-mlx on Apple Silicon; faster: mlx-community/whisper-large-v3-turbo, 'small' elsewhere)")
    ap.add_argument("--no-asr-prompt", action="store_true", help="disable the Warcraft 3 vocabulary prompt for Whisper")
    ap.add_argument("--asr-context", type=int, default=0, help="also feed the last N (0-2) Korean lines to Whisper as context")
    ap.add_argument("--voice", default="en-US-GuyNeural", help="edge-tts voice, e.g. en-US-AriaNeural")
    ap.add_argument("--eleven-voice", default="JBFqnCBsd6RMkjVDRZzb",
                    help="ElevenLabs voice id (used only when ELEVENLABS_API_KEY is set)")
    ap.add_argument("--tts", choices=["edge", "eleven"], default="edge",
                    help="eleven = ElevenLabs via OpenRouter (uses the OpenRouter key), Edge as fallback")
    ap.add_argument("--tts-model", default="elevenlabs/eleven-flash-v2.5", help="OpenRouter speech model for --tts eleven")
    ap.add_argument("--tts-voice", default="george", help="voice name for --tts eleven")
    ap.add_argument("--volume", type=float, default=1.0)
    ap.add_argument("--out-device", default=None, help="output device name/index (default: system output)")
    ap.add_argument("--out-latency", type=float, default=0.4, help="output buffer in seconds (raise if it crackles)")
    ap.add_argument("--translator", choices=["google", "whisper"], default="google",
                    help="google = transcribe Korean then Google Translate (better); whisper = whisper's built-in translate")
    ap.add_argument("--llm-model", default="anthropic/claude-haiku-5.5", help="OpenRouter model slug")
    ap.add_argument("--compare", action="store_true", help="also show Google's translation for each line (testing)")
    ap.add_argument("--max-lag", type=float, default=12.0, help="drop segments older than this many seconds")
    ap.add_argument("--mute", action="store_true", help="don't play audio, just print (testing)")
    ap.add_argument("--save", default=None, help="also write dubbed audio to this .wav")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (testing)")
    a = ap.parse_args()

    log("loading models (first run downloads ~1.5 GB)...")
    asr = ASR(a.model, prompt=not a.no_asr_prompt, ctx_lines=min(a.asr_context, 2))
    log("ASR ready:", asr.name, "(MLX)" if asr.mlx else "(CPU)")
    gt = a.translator == "google"
    okey = load_openrouter_key()
    llm = LLMTranslator(okey, a.llm_model) if okey else None
    log("translator:", f"{a.llm_model} via OpenRouter (Google fallback)" if llm else "Google (no OpenRouter key found)")
    llm_t = []
    import os
    ekey = os.environ.get("ELEVENLABS_API_KEY")
    if a.tts == "eleven" and okey:
        log("TTS:", a.tts_model, "voice", a.tts_voice, "via OpenRouter (Edge fallback)")
    else:
        log("TTS:", "ElevenLabs Flash v2.5" if ekey else "Edge neural voice " + a.voice)
    tts_t = []
    tts_chars = 0

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
            ko = lang == "ko" and len(HANGUL.findall(text)) >= 2 and nsp < 0.6 \
                and not any(j in text for j in JUNK)
            if not ko:
                if text: log(f"  - skip [{lang}] {text[:70]}")
                continue
            en = None
            if gt:
                en = None
                if llm:
                    t0 = time.time()
                    try:
                        en = llm(text)
                        if HANGUL.search(en or ""): en = None
                        llm_t.append(time.time() - t0)
                    except Exception as e:
                        log(f"  (LLM {type(e).__name__}, falling back to Google)")
                cmp_t = 0.0
                if a.compare and en:
                    t0 = time.time()
                    try: g = google_translate(text)
                    except Exception as e: g = f"<{type(e).__name__}>"
                    cmp_t = time.time() - t0
                    log(f"  google [{cmp_t:.2f}s]: {g}")
                if not en: en = translate_text(text)
            if not en: en = asr.translate(seg)
            if not en or HANGUL.search(en) or not re.search(r"[A-Za-z]", en):
                log(f"  - translation failed for: {text[:60]}"); continue
            backlog = play_q.qsize() + seg_q.qsize()
            rate = f"+{min(10 + 15 * backlog, 50)}%"
            t_tts = time.time()
            try:
                if a.tts == "eleven" and okey:
                    try: pcm = openrouter_tts_pcm(en, a.tts_model, a.tts_voice, okey)
                    except Exception as e:
                        log("openrouter tts failed:", type(e).__name__); pcm = tts_pcm(en, a.voice, rate)
                elif ekey:
                    try: pcm = eleven_pcm(en, a.eleven_voice, ekey)
                    except Exception as e:
                        log("elevenlabs failed:", type(e).__name__); pcm = tts_pcm(en, a.voice, rate)
                else:
                    pcm = tts_pcm(en, a.voice, rate)
            except Exception as e:
                log("edge-tts failed:", type(e).__name__)
                if platform.system() != "Darwin": continue
                pcm = say_pcm(en)
            tts_t.append(time.time() - t_tts); tts_chars += len(en)
            play_q.put(pcm)
            d = time.time() - t_seg - (cmp_t if gt else 0)
            lat.append(d)
            log(f"KO: {text}\n         EN: {en}   (ready {d:.1f}s after speech ended)")
    except KeyboardInterrupt:
        pass
    stop.set()
    if llm_t: log(f"LLM call time: median {np.median(llm_t):.2f}s, max {max(llm_t):.2f}s; "
                  f"tokens in/out {llm.tokens_in}/{llm.tokens_out}, cost ${llm.cost:.5f} for {len(llm_t)} lines")
    if tts_t: log(f"TTS time: median {np.median(tts_t):.2f}s, max {max(tts_t):.2f}s, "
                  f"{tts_chars} chars spoken")
    if lat: log(f"latency (speech end -> English audio ready): median {np.median(lat):.1f}s, max {max(lat):.1f}s, n={len(lat)}")
    if SPEAKER: log(f"audio: {SPEAKER.underflows} underflows/dropouts in {SPEAKER.callbacks} output callbacks")
    time.sleep(1.5)


if __name__ == "__main__":
    main()
