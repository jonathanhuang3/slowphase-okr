# slowphase-okr

Manual slow-phase annotation for **OKR gain** estimation. Mark upward slow phases on an elevation trace with two clicks; the tool fits a line through each segment and exports slopes and gain to Excel.

Designed for vestibular/OKR labs using Unity gaze exports (`rotatedGaze.txt` + `gazeTime.txt`).

## Install

```bash
cd slowphase-okr
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
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

## Workflow

1. **Browse gaze file** — `rotatedGaze.txt` (or compatible `(x, y, z)` tuple format)
2. **Browse time file** — `gazeTime.txt` (one timestamp per line, seconds)
3. Enter **Trial ID** and **stimulus velocity** (default 31 deg/s; press Enter to apply after editing)
4. Click **Load trial**
5. **Click twice** on the elevation trace: start and end of each upward slow phase (snaps to nearest sample)
6. **Scroll** to zoom the time axis; **←/→** to pan; use **5 s / 10 s / Full / Reset** view buttons
7. Press **Accept segment** (`A`) to keep the fit
8. **Export Excel** when done

Hover over the plot to see time and elevation at the nearest sample. Press **`?`** (or the **?** button) for the full shortcut list.

## Features

| Feature | Description |
|---------|-------------|
| **Analysis window** | Full trial — first timestamp to last timestamp |
| **Segment list** | Panel showing #, times, gain, R², upward flag; click to select |
| **Segment labels** | `#N` badges on accepted segments in the plot |
| **Edit segments** | Delete selected (`Del`), undo last (`U`), nudge boundaries (`[ ]` start, `, .` end) |
| **Stimulus velocity** | Editable anytime; Enter recalculates gains for all segments |
| **JSON autosave** | Saves to `{trial_id}_slowphase_okr_autosave.json` in the trial folder; offers restore on reload |
| **Reload guard** | Confirms before discarding accepted or pending segments |

**R²** is displayed and written to Excel but segments are **not** auto-rejected.

**Upward OKR:** negative slopes are flagged (`direction_upward = false`) but still exportable.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| Click × 2 | Mark start, then end of slow phase |
| `A` | Accept pending segment |
| `Esc` | Clear pending segment |
| `U` | Undo last accepted segment |
| `Del` | Delete selected segment |
| `←` / `→` | Pan view 1 s |
| Scroll | Zoom time axis |
| `[` / `]` | Nudge selected segment start earlier / later |
| `,` / `.` | Nudge selected segment end earlier / later |
| `Enter` | Apply stimulus velocity (when editing the field) |
| `?` | Open shortcuts help |

## Data format

### USH2A / Unity (GUI)

- `rotatedGaze.txt` — lines like `(x, y, z)`
- `gazeTime.txt` — one float per line (seconds)

Elevation uses the same spherical convention as the USH2A MATLAB helper `unityGazeDirection` (`azimuth = atan2(x,z)`, `elevation = asin(y/r)`).

### CSV (programmatic only)

A CSV with `time` and `elevation` columns can be loaded via `load_csv_trial()` in code; the GUI uses the USH2A pair format.

## Excel output

| Sheet | Contents |
|-------|----------|
| `segments` | One row per accepted segment: indices, times, slope, gain, R², `direction_upward`, stimulus velocity |
| `trial_summary` | `median_gain`, segment count, source file paths, software version |

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
