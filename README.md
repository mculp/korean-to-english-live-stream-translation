# kodub: live Korean → English stream dubber

kodub listens to a live stream (SOOP/AfreecaTV, or anything [streamlink](https://streamlink.github.io/) supports),
picks out **Korean speech only**, translates it, and speaks the English through your speakers. English speech and
game sounds (SFX, music, English unit voices) are skipped, so you keep watching the stream in your browser and hear
an English voice-over for the Korean caster.

Two scripts:

| Script | Translation | Voice |
|---|---|---|
| `kodub_llm.py` (recommended) | Claude Haiku 5.5 via OpenRouter, with context of the last 5 lines and a Warcraft 3 glossary; falls back to Google → MyMemory | Edge neural TTS (free), or ElevenLabs via OpenRouter with `--tts eleven` |
| `kodub.py` (no API key needed) | Google Translate (free web endpoint) → MyMemory fallback | Edge neural TTS, or ElevenLabs if `ELEVENLABS_API_KEY` is set |

## Requirements

- macOS on Apple Silicon (Whisper runs on the GPU via `mlx-whisper`). Other platforms fall back to `faster-whisper` on CPU (`small` model) and are untested.
- [uv](https://docs.astral.sh/uv/). Each script declares its own dependencies inline (PEP 723), so `uv run` installs everything.
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
`ELEVENLABS_API_KEY` is only read from the environment. `.env` is git-ignored; never commit keys.

## Run

```sh
# LLM translation + Edge voice
uv run kodub_llm.py https://play.sooplive.com/<channel>/<broadcast-id>

# LLM translation + ElevenLabs voice through OpenRouter (same key)
uv run kodub_llm.py https://play.sooplive.com/<channel>/<broadcast-id> --tts eleven

# Free version (Google Translate, no key)
uv run kodub.py https://play.sooplive.com/<channel>/<broadcast-id>
```

Press Ctrl+C to stop; it prints latency, cost and audio-underflow stats on exit. Turn the browser's stream volume down a bit so the English is easy to hear.

### Main flags

Both scripts:
- `--quality worst`: stream quality to pull (audio is identical; `worst` saves bandwidth)
- `--voice en-US-GuyNeural`: Edge TTS voice (`uv run --with edge-tts edge-tts --list-voices`)
- `--volume 1.0`, `--out-device <name|index>`, `--out-latency 0.4` (raise if audio crackles)
- `--translator google|whisper`: `whisper` uses Whisper's built-in translate (offline, lower quality)
- `--max-lag 12`: drop speech segments that are more than N seconds behind
- `--eleven-voice <voice_id>`: voice for direct ElevenLabs (needs `ELEVENLABS_API_KEY`)
- `--mute --duration 60 --save out.wav`: test mode (print only, stop after N s, write the dub to a wav)

`kodub.py` only:
- `--model`: Whisper model (default `mlx-community/whisper-large-v3-turbo`)

`kodub_llm.py` only:
- `--asr-model` / `--model`: Whisper model (default `mlx-community/whisper-large-v3-mlx`; `mlx-community/whisper-large-v3-turbo` is faster but less accurate)
- `--no-asr-prompt`: disable the Warcraft 3 vocabulary prompt fed to Whisper
- `--asr-context 0..2`: also feed the last N Korean lines to Whisper
- `--llm-model anthropic/claude-haiku-5.5`: any OpenRouter chat model slug
- `--compare`: also print Google's translation per line (for evaluation)
- `--tts edge|eleven`, `--tts-model elevenlabs/eleven-flash-v2.5`, `--tts-voice george`

## How it works

```
streamlink → PyAV decode → 16 kHz mono float32
  → Silero VAD segmenter (cuts speech segments, ≤12 s)
  → Whisper (MLX) transcribe + language ID
  → filter: language == ko, ≥2 Hangul chars, low no-speech prob, not a known hallucination/prompt echo
  → translate (Haiku via OpenRouter with context → Google → MyMemory → Whisper translate)
  → TTS (ElevenLabs via OpenRouter / ElevenLabs direct / Edge / macOS `say`)
  → Speaker: one persistent output stream at the device's native rate, resampled once, with fades + soft limiter
```

Stages run in threads connected by queues (audio input, segmenter, player); ASR, translation and TTS run in the main loop.
Typical delay is ~3–3.5 s from the end of a Korean sentence to English audio being ready.

## Cost (kodub_llm.py)

- Haiku translation: roughly $0.04/hour of busy commentary.
- ElevenLabs Flash v2.5 via OpenRouter: about $20 per million characters (≈ $0.5–0.9/hour).

## Known limitations

- English lags the stream by a few seconds.
- Game audio is skipped only because it is non-speech or non-Korean. Korean in-game voice lines would be dubbed.
- Whisper still mishears Korean names and slang; garbled transcripts lead to guessed translations. The glossary and ASR prompt are tuned for Warcraft 3.
- Short game sounds are occasionally mistaken for Korean speech.
- Google's free endpoint rate-limits after heavy use; MyMemory (the fallback) is lower quality and has a daily quota.
- Only tested on macOS Apple Silicon.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
