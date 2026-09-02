# Talking Sorting Hat

Interactive Sorting Hat replica: wake word → camera snapshot described locally
(privacy-preserving) → voice conversation with a Claude-powered persona →
spoken reply with synced mouth movement. See [hp_hat_description.md](hp_hat_description.md)
for the full hardware/software spec (in Russian).

Hardware (servos, camera, mic) is still in transit, so development happens on
a Mac first, with every hardware dependency behind an interface (`Camera`,
`ServoController`, `Synthesizer`) so the Raspberry Pi port is a config change.

## Setup

```bash
python3 -m venv .pottervenv
source .pottervenv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY and ELEVENLABS_API_KEY
```

## Running

```bash
python -m hat.chat                        # text-only persona REPL, no audio
python -m hat.main --no-wake --no-vision --text   # full orchestrator over stubs
python -m hat.main                         # the real thing, once all subsystems land
```

## Package layout

- `hat/audio/`, `hat/wake/`, `hat/stt/` — microphone capture, wake word, speech-to-text
- `hat/tts/`, `hat/motion/`, `hat/speech.py` — text-to-speech, lip-sync, servo control
- `hat/brain/`, `hat/vision/`, `hat/main.py` — Claude persona, local vision description, orchestrator
- `hat/config.py` — all settings, `.env`-driven

## Raspberry Pi deployment

Not fully written yet, but the servo/camera/vision pieces are bench-tested:
`cp .env.pi.example .env` on the Pi for the PROFILE=pi settings (PCA9685
servos, rpicam-jpeg camera, Ollama over LAN), then
`sudo apt install -y libportaudio2` (required for `sounddevice`/audio
playback to import at all on Linux -- see requirements-pi.txt) followed by
`pip install -r requirements.txt -r requirements-pi.txt`. Still missing:
full Pi OS setup docs, I2C enable, USB sound card, Bluetooth speaker
pairing, and a systemd service.
