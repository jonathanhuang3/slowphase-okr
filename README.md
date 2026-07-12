# slowphase-okr

Manual slow-phase annotation for **OKR gain** estimation. Mark upward slow phases on an elevation trace with two clicks. The tool fits a line through each segment and exports slopes and gain to Excel.

## Getting started

You need **Python 3.9 or newer** on your computer. Everything else (matplotlib, pandas, and so on) is installed automatically in the steps below.

### Step 1 — Install Python (if you do not have it yet)

**Check if Python is already installed**

Open a terminal and run:

```bash
python3 --version
```

On Windows, try:

```powershell
py --version
```

If you see a version number like `Python 3.11.8`, you can skip to [Step 2](#step-2--get-the-code).

---

**macOS**

1. Download the macOS installer from [python.org/downloads](https://www.python.org/downloads/)
2. Open the `.pkg` file and follow the prompts (use the default options)
3. Open **Terminal** (search for it in Spotlight)
4. Run `python3 --version` to confirm it worked

---

**Windows**

1. Download the Windows installer from [python.org/downloads](https://www.python.org/downloads/)
2. Run the installer
3. On the first screen, check **Add python.exe to PATH** (important)
4. Click **Install Now**
5. Open **PowerShell** or **Command Prompt** (search in the Start menu)
6. Run `py --version` to confirm it worked

If `py` does not work, try `python --version`. If neither works, reinstall Python and make sure **Add python.exe to PATH** is checked.

---

**Linux**

Install Python 3 and tkinter with your package manager. On Ubuntu:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-tk
```

### Step 2 — Get the code

**Option A — Download ZIP (no Git required)**

1. On the GitHub repo page, click **Code** → **Download ZIP**
2. Unzip the folder
3. Note where it was saved (for example `Desktop/slowphase-okr`)

**Option B — Clone with Git**

If you have [Git](https://git-scm.com/downloads) installed:

```bash
git clone https://github.com/jonathanhuang/slowphase-okr.git
cd slowphase-okr
```

You can also clone with **GitHub Desktop** (Repository → Clone repository).

### Step 3 — Install slowphase-okr

Open a terminal in the project folder and install once. Replace the `cd` path with wherever you saved the folder.

**macOS / Linux**

```bash
cd ~/Desktop/slowphase-okr
python3 -m pip install -e .
```

**Windows**

```powershell
cd $env:USERPROFILE\Desktop\slowphase-okr
py -m pip install -e .
```

If `py` is not found, use `python` instead.

This installs the app and its dependencies into your normal Python. You only need to do this once (or again after updating the code).

### Step 4 — Run the app

Open a terminal and run:

```bash
slowphase-okr
```

A window should open. After install, you can run this from any folder — you do not need to `cd` into the project each time.

**If `slowphase-okr` does not work**, try:

```bash
python3 -m slowphase_okr
```

On Windows, use `py -m slowphase_okr` if needed.

If nothing happens or you see an error, see [Troubleshooting](#troubleshooting).

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `python3` or `py` not found | Install Python from [python.org](https://www.python.org/downloads/). On Windows, reinstall with **Add to PATH** checked. |
| `pip` not found | Run `python3 -m pip install -e .` (macOS/Linux) or `py -m pip install -e .` (Windows) from the project folder |
| `No module named '_tkinter'` (Linux) | Run `sudo apt install python3-tk` (Ubuntu) or install the tk package for your distro |
| App opens then closes immediately | Run `python3 -m slowphase_okr` (or `py -m slowphase_okr` on Windows) from the terminal to see the error message |
| `slowphase-okr` command not found | Try `python3 -m slowphase_okr` / `py -m slowphase_okr`, or re-run Step 3 (`pip install -e .`) |

## Workflow

### Manual annotation

1. **Browse gaze file** — your gaze direction file (see [Data format](#data-format))
2. **Browse time file** — your timestamp file (one time per gaze sample)
3. **Browse OKR log** *(optional)* — stimulus event log with block and fixation timing (see [OKR log](#okr-log-optional))
4. Enter **Trial ID** and **stimulus velocity** (default 31 deg/s, press Enter to apply after editing)
5. Click **Load trial**
6. Choose **Signal** — elevation (default) or SRanipal pupil X/Y if those files are in the trial folder
7. **Click twice** on the trace: start and end of each upward slow phase (snaps to nearest sample)
8. **Scroll vertically** or use **←/→** to pan the time axis (20% of window per step), and pick a **Window** (1 s, 2 s, 5 s, 10 s, or Full trial)
9. Press **Accept segment** (`A`) to keep the fit
10. **Export Excel** when done

### Semi-automatic annotation (recommended with OKR log)

1. Complete steps 1–5 above (include OKR log if available).
2. Adjust [auto-detect settings](#auto-detect-optional) — hover any parameter in the app for a short explanation.
3. Click **Propose segments** — candidates appear as **?** (blue dashed lines).
4. Use **N** / **P** (or the buttons) to step through proposals; **A** to accept, **Del** to reject.
5. Hover near a segment edge to highlight it, then drag to adjust; or nudge with `[` `]` `,` `.` if needed; manually click any slow phases the detector missed.
6. **Export Excel** when done (only **✓** accepted segments are exported).

**File pairing:** use `rotatedGaze.txt` with `gazeTime.txt` from the same folder (not `sranipalGazeTime.txt` unless you also use `sranipalGazeSpace.txt`).

Hover over the plot to see time and elevation at the nearest sample. Press **`?`** (or the **Help** button) for the full shortcut list.

## Features

| Feature | Description |
|---------|-------------|
| **Analysis window** | Full trial — first timestamp to last timestamp |
| **Signal source** | Elevation from rotated gaze, or SRanipal pupil X/Y (auto-loaded from trial folder) |
| **Zero elev at start** | Optional (on by default). Subtracts elevation at the first valid sample so the trace starts at 0° — removes headset pose offset. Slopes and gains are unchanged. |
| **Fixed y-axis** | Signal scale stays fixed to the full trial while you pan in time |
| **Segment list** | Shows proposed (`?`) and accepted (`✓`) segments sorted by time |
| **Segment labels** | `#N` on accepted (green), `?N` on proposed (blue) segments in the plot |
| **Edit segments** | Delete selected (`Del`), undo last (`U`), drag edges on plot, nudge boundaries (`[ ]` start, `, .` end) |
| **Auto-detect** | Sliding-window detector proposes slow phases for manual review (see below) |
| **Stimulus velocity** | Editable anytime. Enter recalculates gains for all segments. |
| **JSON autosave** | Saves accepted segments to `{trial_id}_slowphase_okr_autosave.json`; restore on reload |
| **Reload guard** | Confirms before discarding accepted or proposed segments |
| **OKR log markers** | Optional upload marks contrast-block starts (purple) and fixation-cross starts (gray) |
| **Condition readout** | With OKR log loaded, shows contrast, direction, flicker/persistent, and session Increment/Decrement for the hovered (or view-center) time |
| **Invalid gaze samples** | `(NaN, NaN, NaN)` lines stay aligned but are skipped for clicking and fitting |
| **Parameter tooltips** | Hover auto-detect settings in the app for plain-language help |

**R²** is displayed and written to Excel but segments are **not** auto-rejected.

**Upward OKR:** negative slopes are flagged (`direction_upward = false`) but still exportable.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| Click × 2 | Mark start, then end of slow phase (manual) |
| `A` | Accept pending manual segment (priority), hovered proposed segment, or selected proposed |
| `N` / `P` | Next / previous proposed segment (pans plot to it) |
| `U` | Undo last accepted segment |
| `Del` | Delete selected accepted or proposed segment |
| `←` / `→` | Pan view by 20% of visible window |
| Scroll (vertical) | Pan time forward / backward (20% of visible window per tick) |
| `Esc` | Cancel boundary drag or clear pending manual segment |
| `[` / `]` | Nudge selected segment start earlier / later |
| `,` / `.` | Nudge selected segment end earlier / later |
| Drag edges | Select a segment, hover near start/end until the edge highlights, then drag (snaps to nearest data point) |
| `Enter` | Apply stimulus velocity (when editing the field) |
| `?` | Open shortcuts help |

## Auto-detect (optional)

Conservative semi-automatic mode: the detector **proposes** segments; you **accept** or **delete** each one. Proposals are not exported until accepted.

**How it works:** averages duplicate Unity timestamps, slides a short window along the elevation trace, fits a line in each window, merges nearby hits, optionally expands boundaries for a better R², and filters saccades.

Hover any setting in the **Auto-detect** panel for a tooltip. Summary:

| Parameter | Default | What it does |
|-----------|---------|--------------|
| **Direction** | Auto | **Auto:** Up/Down from each OKR contrast block. **Up/Down:** force one direction for the whole search. Auto requires OKR log + contrast-block checkbox. |
| **Max saccade velocity** | 100 deg/s | Reject candidates with a speed spike above this inside the segment (filters saccades). Lower = stricter. |
| **Min duration** | 50 ms | Shortest segment to keep. Shorter = more proposals. |
| **Min R²** | 0.75 | Minimum straightness of the line fit (0–1). Lower = accept noisier traces. |
| **Window** | 100 ms | Sliding fit window length. Typical slow phase is longer than this; window should be 80–150 ms. |
| **Merge gap** | 40 ms | Join fragment proposals separated by less than this gap. |
| **Refine boundaries** | on | Expand start/end while R² stays good. |
| **Only inside contrast blocks** | off | Search only during stimulus blocks from OKR log (not fixation). Enable when using OKR log. |

**Suggested starting point with OKR log:** Direction **Auto**, contrast blocks **on**, Window **100 ms**, Merge gap **40 ms**, Min R² **0.7–0.75**.

## Data format

The GUI expects **two text files per trial**: one with gaze direction and one with timestamps. Filenames do not matter — use whatever your pipeline exports.

### Gaze + timestamp files (GUI)

This is the format the app loads.

| File | What it contains |
|------|------------------|
| **Gaze file** | One 3D gaze direction per line, as `(x, y, z)` tuples |
| **Time file** | One timestamp per line, in seconds, aligned sample-for-sample with the gaze file |

**Example filenames** (from our Vive Pro Eye / Tobii + Unity setup): `rotatedGaze.txt` and `gazeTime.txt`, or raw SRanipal exports like `sranipalGazeSpace.txt` and `sranipalGazeTime.txt`. Your lab may use different names — that is fine as long as the content matches.

This tool was developed for our lab to support gaze data recorded with a **HTC Vive Pro Eye** headset (Tobii eye tracking). The gaze file holds direction vectors in the experiment’s rotated coordinate frame. The app converts `(x, y, z)` to elevation (`azimuth = atan2(x,z)`, `elevation = asin(y/r)`).

**Requirements**

- The same number of lines in both files (one timestamp per gaze line, including invalid samples)
- Timestamps in seconds
- Gaze lines as `(x, y, z)` tuples. Use `(NaN, NaN, NaN)` for missing tracking — these rows stay aligned with timestamps but are treated as invalid (no elevation, not clickable)

**Duplicate timestamps:** Unity `gazeTime.txt` often repeats the same time for several gaze samples (one Unity frame, multiple eye samples). That is normal. Auto-detect averages elevation at each unique time before searching.

**Pair the right files:** `rotatedGaze.txt` + `gazeTime.txt` (same line count). Do not mix `rotatedGaze.txt` with `sranipalGazeTime.txt`.

**SRanipal pupil position (optional):** If present in the same trial folder, the app also loads `sranipalLeftPupilPositions.txt` / `sranipalRightPupilPositions.txt` with matching `*PupilPositionTimes.txt` (or `*PupilTimes.txt`). Pupil coordinates are normalized screen units `(x, y)` per line. Use the **Signal** menu to plot pupil Y (vertical OKR) or pupil X instead of elevation. Segment marking, auto-detect, and gain (`slope / stimulus velocity`) work the same way; pupil slopes are in normalized units per second. The viewing eye is inferred from the OKR log `eyePatch` column (patched eye) or trial name (`RE` / `LE`).

### OKR log (optional)

Older stimulus versions did not write this file. If your session includes an OKR condition log, you can upload it to show stimulus timing on the plot.

| File | What it contains |
|------|------------------|
| **OKR log** | Tab-separated event table from Unity (e.g. `OKR_Log_Patient_Testing.txt`) |

**Typical columns:** `eventIndex`, `eventType`, `contrastBlockIndex`, `startTime`, `endTime`, and others. Comment lines start with `#`.

**Time base:** `startTime` / `endTime` use Unity `Time.time` (seconds since Play), which should match your gaze timestamp file.

**How events are classified:**

| Event name contains | Plot marker |
|---------------------|-------------|
| `fixation` (case-insensitive) | Gray dotted line — fixation cross / ITI start (`F`) |
| Anything else | Purple dashed line — contrast block start (label e.g. `B2↓`) |

Block labels use `contrastBlockIndex` and direction when available. New block event names are detected automatically without updating the app.

**Condition readout:** With an OKR log loaded, a line under the plot shows the condition at the hovered time (or the center of the current view): contrast level, direction, flicker vs persistent, and session tags from `StimulusName` (e.g. Increment / Decrement, White / Black dots).

**Example filenames:** `OKR_Log_Patient_Testing.txt` or similar in the same folder as gaze/time files.

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
