# kodub: live Korean or Mandarin → English stream dubber

kodub listens to a live stream, picks out speech in one source language, translates it, and speaks the English through your speakers. English speech and game sounds are skipped, so in the normal mode you keep the stream open in your browser (volume down a bit) and hear an English voice-over.

Mandarin Warcraft 3 on Douyu, Huya and Douyin uses `--lang zh`. Korean (the default) is unchanged.

Two scripts:

| Script | Translation | Voice |
|---|---|---|
| `kodub_llm.py` (recommended) | Claude Haiku 5.5 via OpenRouter, with the last 5 lines, a Warcraft 3 glossary, and optional match context; falls back to Google → MyMemory | Edge neural TTS (free), or ElevenLabs Flash v2.5 via OpenRouter with `--tts eleven` |
| `kodub.py` (no API key needed) | Google Translate → MyMemory fallback, with `--lang ko\|zh` | Edge neural TTS, or ElevenLabs if `ELEVENLABS_API_KEY` is set |

`--context`, the caster glossary, and `--overdub` are on `kodub_llm.py` only.

## Watch a Chinese Warcraft 3 stream

Streamlink is not installed globally. `uv run` installs it (8.5 or newer, which has Douyu, Huya and Douyin plugins) into the script environment. Douyu room URLs need the numeric id (`www.douyu.com/122024`).

```sh
cd ~/kodub

# LioN, semi-pro. Prints ZH lines and speaks English. Turn the browser down a little.
uv run kodub_llm.py "https://www.douyu.com/122024" --lang zh --context "LioN casting semi-pro Warcraft 3"

# Fly100%
uv run kodub_llm.py "https://www.douyu.com/255329" --lang zh --context "Fly100% Warcraft 3"

# eer0. If the title starts with 录播 the room is a replay, not a live cast.
uv run kodub_llm.py "https://www.douyu.com/1786555" --lang zh --context "eer0 Warcraft 3"

# Huya
uv run kodub_llm.py "https://www.huya.com/642681" --lang zh

# Douyin, only while the room is actually live (streamlink status 2). An offline room has a title and no media.
uv run kodub_llm.py "https://live.douyin.com/838580843877" --lang zh --context "Warcraft 3"
```

Nothing else to install on the Mac if `uv` is already there and `OPENROUTER_API_KEY` is in `~/kodub/.env`. The first run on a new machine downloads the Whisper model; a machine that already dubbed Korean reuses it. Audio from these rooms is FLV, which kodub detects. `--quality worst` (the default) pulls the lowest Douyu bitrate, often `900k`.

Low-latency voice, same OpenRouter key: add `--tts eleven`. That is ElevenLabs Flash v2.5 with the voice George (`--tts-voice george`), one consistent caster-style voice.

```sh
uv run kodub_llm.py "https://www.douyu.com/122024" --lang zh --tts eleven \
  --context "LioN casting semi-pro Warcraft 3"
```

### One mixed soundtrack (`--overdub`)

Normal mode does not play the stream. The browser stays live and the English arrives a few seconds after the caster finishes each line.

`--overdub` plays the stream audio itself, held long enough for the English to land on the line it translates, and ducks that audio (caster and game sounds together, since they are one mix) while the English voice speaks. Mute the browser or you will hear the caster twice. The picture in the browser stays live, so the mix lags it.

```sh
uv run kodub_llm.py "https://www.douyu.com/122024" --lang zh --tts eleven --overdub \
  --context "LioN casting semi-pro Warcraft 3"
```

The hold defaults to 7 seconds (`--overdub-delay`). Add `--out-latency` (default 0.4s) for the output buffer. If a line is logged as overdub-late, raise `--overdub-delay`. `--duck 0.22` is how loud the original mix stays under the English voice.

## Requirements

- macOS on Apple Silicon (Whisper runs on the GPU via `mlx-whisper`). Other platforms fall back to `faster-whisper` on CPU (`small` model) and are untested for real listening.
- [uv](https://docs.astral.sh/uv/). Each script declares its own dependencies inline (PEP 723), so `uv run` installs everything, including streamlink. Do not install streamlink yourself.
- Python 3.11–3.12 (uv picks it automatically).
- Internet access. The first run downloads the Whisper model (~3 GB for large-v3, ~1.6 GB for turbo).
- For `kodub_llm.py`: an [OpenRouter](https://openrouter.ai) API key.

## Setup

```sh
git clone https://github.com/mculp/korean-to-english-live-stream-translation.git ~/kodub
cd ~/kodub
cp .env.example .env            # then put your key after OPENROUTER_API_KEY=
```

`kodub_llm.py` reads `OPENROUTER_API_KEY` from the environment, or else from `~/kodub/.env`, `~/.env` or `~/.hermes/.env`.
`ELEVENLABS_API_KEY` is only read from the environment (direct ElevenLabs, not the OpenRouter voice path). `.env` is git-ignored; never commit keys.

## Run (Korean)

```sh
# LLM translation + Edge voice
uv run kodub_llm.py https://play.sooplive.com/<channel>/<broadcast-id>

# LLM translation + ElevenLabs Flash v2.5 through OpenRouter (same key)
uv run kodub_llm.py https://play.sooplive.com/<channel>/<broadcast-id> --tts eleven

# Free version (Google Translate, no key)
uv run kodub.py https://play.sooplive.com/<channel>/<broadcast-id>
```

Press Ctrl+C to stop; it prints latency, cost and audio-underflow stats on exit.

### Main flags

Both scripts:
- `--lang ko|zh`: source language. Default `ko`, so existing Korean commands do not change. `zh` accepts Whisper's Chinese label, counts Han characters instead of Hangul, and asks Google/MyMemory for `zh-CN`.
- `--quality worst`: stream quality to pull (audio is what we need; `worst` saves bandwidth). Douyu qualities look like `900k`.
- `--voice en-US-GuyNeural`: Edge TTS voice (`uv run --with edge-tts edge-tts --list-voices`)
- `--volume 1.0`, `--out-device <name|index>`, `--out-latency 0.4` (raise if audio crackles)
- `--translator google|whisper`: `whisper` uses Whisper's built-in translate (offline, lower quality)
- `--max-lag 12`: drop speech segments that are more than N seconds behind
- `--eleven-voice <voice_id>`: voice for direct ElevenLabs (needs `ELEVENLABS_API_KEY`). Default is George.
- `--mute --duration 60 --save out.wav`: test mode (print only, stop after N s, write the dub to a wav)

`kodub.py` only:
- `--model`: Whisper model (default `mlx-community/whisper-large-v3-turbo`)

`kodub_llm.py` only:
- `--context "Fly100% vs Lyn, Bo3"`: matchup or anything else the translator should trust. The stream title and streamer name are added automatically when streamlink returns them.
- `--asr-model` / `--model`: Whisper model (default `mlx-community/whisper-large-v3-mlx`; `mlx-community/whisper-large-v3-turbo` is faster but less accurate)
- `--no-asr-prompt`: disable the Warcraft 3 vocabulary prompt fed to Whisper
- `--asr-context 0..2`: also feed the last N source-language lines to Whisper
- `--llm-model anthropic/claude-haiku-5.5`: any OpenRouter chat model slug
- `--compare`: also print Google's translation per line (for evaluation)
- `--tts edge|eleven`, `--tts-model elevenlabs/eleven-flash-v2.5`, `--tts-voice george`
- `--overdub`, `--overdub-delay 7`, `--duck 0.22`: delayed stream mix. See above. Ignored when `--mute` is set.

## How it works

```
streamlink (MPEG-TS or FLV) → PyAV decode → 16 kHz mono float32
  → Silero VAD segmenter (cuts speech segments, ≤12 s)
  → Whisper (MLX) transcribe + language ID
  → filter: language matches --lang, ≥2 Hangul or Han characters, low no-speech prob,
    not a known hallucination / prompt echo / subtitle credit
  → translate (Haiku via OpenRouter with the last 5 lines, glossary, and match context
    → Google → MyMemory → Whisper translate). The model may answer [skip] for noise.
  → TTS (ElevenLabs Flash v2.5 via OpenRouter / ElevenLabs direct / Edge / macOS `say`)
  → Speaker, or with --overdub a delay line that ducks the stream under the English
```

Stages run in threads connected by queues. ASR, translation and TTS run in the main loop.

### Latency

Normal mode, on Apple Silicon with the default large-v3 model, Haiku, and Edge: about 3–3.5 seconds from the end of a line until the English audio is ready. That number is the one this project already measured on a Mac. A Linux check against LioN's live Douyu room, with faster-whisper `small` on CPU, Google (no OpenRouter key), and Edge, was faster than that Mac figure: median 2.6s from speech end to English ready (8 lines, max 4.0s), of which Edge TTS was median 0.45s. Treat the Mac large-v3 + Haiku number as the one to expect at the laptop. `--tts eleven` (Flash v2.5, voice George) is the low-latency voice; Flash's model time is a small fraction of a second, and Edge's round trip is usually the slower TTS. The browser picture stays live. There is no extra hold.

Overdub mode lags the live picture by `--overdub-delay` (default 7s) plus the output buffer (`--out-latency`, default 0.4s, and whatever Core Audio actually reports). The English is scheduled at the start of the original line inside that delayed mix. A line fits when its length plus the processing time above stays under the hold. Longer lines start the English late; the log says by how much. On exit, kodub prints the processing time it actually measured and the overdub lag it is using.

## Cost (kodub_llm.py)

- Haiku translation: roughly $0.04/hour of busy commentary. The Chinese system prompt is longer than the Korean one, so a Mandarin stream costs a bit more per line.
- ElevenLabs Flash v2.5 via OpenRouter: about $20 per million characters (≈ $0.5–0.9/hour).

## Known limitations

- English lags the stream by a few seconds in normal mode, and by the overdub hold when `--overdub` is on. The browser video is not delayed.
- Overdub ducks the whole stream mix, game audio included, because the caster and the game are not separate tracks.
- Game audio is skipped in normal mode only because it is non-speech or not the source language. Source-language in-game voice lines would be dubbed.
- Whisper still mishears names and slang. The glossary and ASR prompt are tuned for Warcraft 3 (Korean, and Mandarin as casters say it, including the Forsaken Paladin).
- Short game sounds are occasionally mistaken for speech.
- Google's free endpoint rate-limits after heavy use; MyMemory is lower quality and has a daily quota.
- Douyin returns no media unless the room is actually live. A room page can still show a title while streamlink reports it offline.
- Only tested for listening on macOS Apple Silicon. Container sniffing and the Mandarin filters were checked on Linux; MLX, the speakers, and a real overdub listen were not.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
