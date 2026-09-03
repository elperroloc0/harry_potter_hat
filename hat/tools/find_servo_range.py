"""Interactive hunt for one servo's real duty-cycle limits, by eye.

    python -m hat.tools.find_servo_range --channel 15

Single keypresses, no Enter. Nudge the horn until it is exactly where you
want it, mark that spot, and repeat for the other end; the tool prints the
numbers to paste into ServoCal at the end.

Why this exists rather than trusting the datasheet: on this rig the
"standard" 1.0/1.5/2.0 ms triple is neither centred nor symmetric once a
horn is pressed onto the splines -- the horn's own zero lands wherever the
spline teeth allow, usually a good few degrees off -- and the servo also
free-runs well before the nominal end of travel. Both have to be found by
looking at the thing.

Start with the servo de-energized (it is, on launch) and set the horn
horizontal by hand. That is your reference: step until the powered horn
returns to that same horizontal, and mark it as the centre.
"""

from __future__ import annotations

import argparse
import sys
import termios
import tty

# Wide enough to find where free-running starts, but not so wide as to sit
# on a hard stall for long. Past its tracking range a servo does not stop --
# it turns continuously - so expect that while hunting, and step back in.
HARD_MIN_PCT = 2.5
HARD_MAX_PCT = 12.5

KEYS = """
  arrows / + -   nudge the horn (left = lower duty)
  [ ]            step size down / up
  c              mark this spot as CENTRE (horizontal)
  n              mark this spot as the LOW limit
  x              mark this spot as the HIGH limit
  0              cut the signal (horn goes limp, safe to reposition by hand)
  q              finish and print the numbers
"""


def _read_key() -> str:
    """One keypress, no Enter. Arrow keys arrive as a three-byte escape."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hat.tools.find_servo_range")
    parser.add_argument("--channel", type=int, default=15, help="PCA9685 channel (default 15, the mouth)")
    parser.add_argument("--freq", type=int, default=50, help="PWM frequency in Hz (default 50)")
    parser.add_argument("--start", type=float, default=7.5, help="duty cycle to begin at, in percent (default 7.5)")
    parser.add_argument("--step", type=float, default=0.25, help="initial nudge size, in percent (default 0.25)")
    args = parser.parse_args(argv)

    import board  # type: ignore[import-not-found]
    import busio  # type: ignore[import-not-found]
    from adafruit_pca9685 import PCA9685  # type: ignore[import-not-found]

    pca = PCA9685(busio.I2C(board.SCL, board.SDA))
    pca.frequency = args.freq
    channel = pca.channels[args.channel]

    pct, step = args.start, args.step
    marks: dict[str, float] = {}
    powered = False

    print(f"Channel {args.channel} at {args.freq} Hz. Servo is OFF.")
    print("Set the horn horizontal by hand now -- that is your reference.")
    print(KEYS)
    print("Press any key to energize.")
    _read_key()

    def apply() -> None:
        nonlocal powered
        channel.duty_cycle = int(pct / 100 * 65535)
        powered = True

    apply()
    try:
        while True:
            ms = pct / 100 / args.freq * 1000
            flags = " ".join(f"{k}={v:.2f}%" for k, v in sorted(marks.items()))
            state = f"{pct:6.2f}%  ({ms:.2f} ms)  step {step:.2f}" if powered else "   OFF   "
            print(f"\r  {state}   {flags}          ", end="", flush=True)

            key = _read_key()
            if key in ("q", "\x03"):
                break
            elif key in ("\x1b[C", "+", "="):
                pct = min(HARD_MAX_PCT, pct + step); apply()
            elif key in ("\x1b[D", "-", "_"):
                pct = max(HARD_MIN_PCT, pct - step); apply()
            elif key == "]":
                step = min(2.0, step * 2)
            elif key == "[":
                step = max(0.05, step / 2)
            elif key == "0":
                channel.duty_cycle = 0; powered = False
            elif key in ("c", "n", "x"):
                marks[{"c": "centre", "n": "low", "x": "high"}[key]] = pct
    finally:
        channel.duty_cycle = 0

    print("\n\nServo off.\n")
    if not marks:
        print("Nothing marked.")
        return 0

    for name, value in sorted(marks.items()):
        print(f"  {name:<7} {value:.2f}%")

    low, high, centre = marks.get("low"), marks.get("high"), marks.get("centre")
    if low is not None and high is not None:
        print("\nPaste into ServoCal in hat/config.py:\n")
        print(f"    min_duty: float = {low / 100:.4f}")
        print(f"    max_duty: float = {high / 100:.4f}")
        if centre is not None:
            print(f"    centre_duty: float = {centre / 100:.4f}")
            span = min(centre - low, high - centre)
            print(
                f"\n  Note: centre sits {centre - low:.2f}% above the low mark and "
                f"{high - centre:.2f}% below the high one."
            )
            if abs((centre - low) - (high - centre)) > 0.05:
                print(
                    f"  That is lopsided, so symmetric travel is limited to the "
                    f"narrower side: +/-{span:.2f}% around centre."
                )
    else:
        print("\nMark both a low (n) and a high (x) limit to get config lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
