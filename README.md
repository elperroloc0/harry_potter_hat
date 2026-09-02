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
pip install -r requirements-audio.txt   # only if working on wake word/STT -- see that file
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

Not fully written yet, but servos/camera/vision/speech are all bench-tested
end to end. `cp .env.pi.example .env` on the Pi for the PROFILE=pi settings
(PCA9685 servos, rpicam-jpeg camera, Ollama over LAN), then:

```bash
sudo apt install -y libportaudio2   # required for sounddevice to import at all
pip install -r requirements.txt -r requirements-pi.txt
```

**Bluetooth speaker audio**: pairing/connecting via `bluetoothctl` alone does
NOT make the speaker usable for playback -- bare ALSA has no native
Bluetooth support. On this rig (Pi OS running PipeWire, confirmed via
`pgrep -a pipewire`), what actually worked was bridging ALSA to PipeWire's
pulse-compatible layer:

```bash
sudo apt install -y libasound2-plugins
cat > ~/.asoundrc << 'EOF'
pcm.!default { type pulse }
ctl.!default { type pulse }
EOF
```

Verify with `paplay /usr/share/sounds/alsa/Front_Center.wav` (should be
audible) and `python -c "import sounddevice; print(sounddevice.query_devices())"`
(should list a `pulse`/`default` device, not hang or error). See
`project_hat_hardware_status` memory for the full troubleshooting path if
this doesn't work out of the box on a different Pi OS image -- `bluealsa`
was tried first and didn't pan out on this setup, so don't assume it's
required before checking whether PipeWire is already handling it.

Still missing: full Pi OS setup docs, I2C enable, USB sound card
(microphone side), and a systemd service.
