"""Bench calibration tool for a real MG90S/SG90 on the PCA9685 -- Pi only.

Bypasses ServoCal/PCA9685Servos entirely and drives one channel directly by
raw duty cycle -- the same mechanism PCA9685Servos itself uses (see
hat/motion/servos.py) -- so you can safely sweep a single servo, find its
real safe duty-cycle range, and write those numbers into ServoCal in
hat/config.py, without trusting any calibration values up front.

Deliberately does NOT go through adafruit_motor.servo / ServoKit's angle
API: on the bench that gave unstable, jittery motion. Raw
PCA9685.channels[n].duty_cycle is what actually worked.

Usage (on the Pi, after `pip install -r requirements-pi.txt`):
    python -m hat.tools.calibrate_servos --channel 15
    python -m hat.tools.calibrate_servos --channel 15 --min-duty 0.005 --max-duty 0.20

At the prompt, type a duty cycle as a percent (e.g. 7.5 for 7.5%) and press
enter to move the servo there. Start near the middle of the safe range
(~7.5%, roughly 90 degrees) and creep outward in small steps (~0.5%) -- stop
at the first sign of buzzing/straining (a mechanical limit) or of the servo
losing tracking and spinning continuously (you've left the electrical range
the PWM signal is interpreted correctly in -- back off immediately, that is
not a limit to push past). Type 'q' to quit.

On MG90S units bench-tested for this project, 1.5%-14% duty cycle was the
safe range across the full 0-180 degree sweep -- see ServoCal.min_duty/
max_duty in hat/config.py.
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channel", type=int, required=True, help="PCA9685 channel (0-15)")
    parser.add_argument("--freq", type=int, default=50, help="PWM frequency, Hz (default 50, matches ServoCal.pca9685_freq_hz)")
    parser.add_argument(
        "--min-duty", type=float, default=0.002, help="lower bound of the duty-cycle range this tool will let you send, as a fraction 0-1 (default 0.002 = 0.2%%, a conservative generic floor)"
    )
    parser.add_argument(
        "--max-duty", type=float, default=0.20, help="upper bound of the duty-cycle range this tool will let you send, as a fraction 0-1 (default 0.20 = 20%%, a conservative generic ceiling)"
    )
    args = parser.parse_args(argv)

    # Deferred: this module must be importable on the Mac even though it can
    # only run on the Pi (same reasoning as PCA9685Servos in motion/servos.py).
    import board  # type: ignore[import-not-found]
    import busio  # type: ignore[import-not-found]
    from adafruit_pca9685 import PCA9685  # type: ignore[import-not-found]

    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = args.freq
    channel = pca.channels[args.channel]

    min_pct, max_pct = args.min_duty * 100, args.max_duty * 100
    print(f"Channel {args.channel}, freq {args.freq} Hz, duty range [{min_pct:.2f}%, {max_pct:.2f}%].")
    print("Enter a duty cycle in percent to move there, 'q' to quit.")
    print("Start near the middle and creep outward in ~0.5% steps; stop at the first sign of")
    print("buzzing/strain (mechanical limit) or the servo spinning continuously instead of")
    print("holding an angle (you've left the valid electrical range -- back off).\n")

    last_pct: float | None = None
    try:
        while True:
            raw = input(f"duty %{f' [last={last_pct}]' if last_pct is not None else ''}> ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                break
            if not raw:
                continue
            try:
                pct = float(raw)
            except ValueError:
                print("not a number, try again")
                continue
            duty = pct / 100.0
            if not (args.min_duty <= duty <= args.max_duty):
                print(f"outside [{min_pct:.2f}%, {max_pct:.2f}%], refusing to send it")
                continue
            channel.duty_cycle = int(duty * 65535)
            last_pct = pct
            print(f"  -> duty_cycle {channel.duty_cycle} ({pct:.2f}%)")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        # Critical: the PCA9685 keeps outputting the last duty cycle it was
        # told, on its own, indefinitely -- exiting this script does not
        # stop it. A stale signal left live is exactly what made the servo
        # lurch/spin on the *next* run before that run had done anything.
        channel.duty_cycle = 0
        print(
            "\nchannel zeroed. Note down the safe min/max duty-cycle percentages "
            "(as fractions 0-1) you found and put them in ServoCal.min_duty/max_duty "
            "(hat/config.py)."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
