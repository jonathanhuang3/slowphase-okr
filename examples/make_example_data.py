"""Generate a small synthetic example trial for manual testing."""

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent


def main() -> None:
    rng = np.random.default_rng(42)
    n = 3600
    times = np.arange(n) * (1.0 / 90.0)
    elev = np.zeros(n)
    for i in range(1, n):
        elev[i] = elev[i - 1] + (28.0 / 90.0) + rng.normal(0, 0.02)
        if i % 400 == 0:
            elev[i] -= rng.uniform(0.5, 1.5)

    y = np.sin(np.radians(elev))
    z = np.cos(np.radians(elev))
    x = np.zeros(n)

    gaze_lines = [f"({x[i]:.6f}, {y[i]:.6f}, {z[i]:.6f})" for i in range(n)]
    (OUT / "example_rotatedGaze.txt").write_text("\n".join(gaze_lines) + "\n")
    (OUT / "example_gazeTime.txt").write_text(
        "\n".join(f"{t:.6f}" for t in times) + "\n"
    )
    print(f"Wrote example files to {OUT}")


if __name__ == "__main__":
    main()
