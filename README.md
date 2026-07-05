# slowphase-okr

Manual slow-phase annotation for **OKR gain** estimation. Mark upward slow phases on an elevation trace with two clicks; the tool fits a line through each segment and exports slopes and gain to Excel.

Designed for vestibular/OKR labs using Unity gaze exports (`rotatedGaze.txt` + `gazeTime.txt`), with a path toward open-source sharing on GitHub.

## Install

From a clone of this repository:

```bash
cd slowphase-okr
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Or install directly from GitHub (once published):

```bash
pip install git+https://github.com/YOUR_USERNAME/slowphase-okr.git
```

**Requirements:** Python 3.9+, tkinter (included with most Python installers; on Linux you may need `python3-tk`).

## Run

```bash
slowphase-okr
```

or

```bash
python -m slowphase_okr
```

## Workflow (v1)

1. **Browse gaze file** — `rotatedGaze.txt` (or compatible `(x, y, z)` tuple format)
2. **Browse time file** — `gazeTime.txt` (one timestamp per line, seconds)
3. Enter **Trial ID** and **stimulus velocity** (default 31 deg/s)
4. Click **Load trial**
5. **Click twice** on the elevation trace: start and end of each upward slow phase (snaps to nearest sample)
6. **Scroll** over the plot to zoom the time axis; **Left/Right arrows** to pan (default **5 s** visible window)
7. Press **Accept segment** (or `A`) to keep the fit; **Undo** (`U`) to remove the last segment
7. **Export Excel** when done — `segments` sheet (per segment) + `trial_summary` sheet (median gain)

**Analysis window:** 0–40 s from the first timestamp in the time file.

**R²** is displayed and written to Excel but segments are **not** auto-rejected.

**Upward OKR:** negative slopes are flagged (`direction_upward = false`) but still exportable.

## Data format

### USH2A / Unity (default)

- `rotatedGaze.txt` — lines like `(x, y, z)`
- `gazeTime.txt` — one float per line (seconds)

Elevation is computed with the same spherical convention as the USH2A MATLAB helper `unityGazeDirection` (`azimuth = atan2(x,z)`, `elevation = asin(y/r)`).

### CSV (future-friendly loader in code)

A CSV with columns `time` and `elevation` (degrees) can be loaded programmatically via `load_csv_trial()`; the GUI v1 uses the USH2A pair format.

## Excel output

| Sheet | Contents |
|-------|----------|
| `segments` | One row per accepted segment: indices, times, slope, gain, R², `direction_upward` |
| `trial_summary` | `median_gain`, segment count, source file paths, software version |

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Roadmap (post-v1)

- JSON autosave, velocity subplot, configurable analysis window (0.5 s skip)
- Automated segment overlay for validation
- Multi-rater / ICC workflow
- Batch trial queue

See `Analysis/PostExperiment/Jonathan/OKR_Manual_Annotator_Planning.docx` for full planning notes.

## Citation

If you use this tool, cite the GitHub repository URL and version (shown in Excel exports as `software_version`).

## License

MIT — see [LICENSE](LICENSE).
