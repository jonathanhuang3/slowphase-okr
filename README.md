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

**PowerShell note:** If activating the virtual environment later shows a security error, use **Command Prompt** instead, or run this once in PowerShell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

**Linux**

Install Python 3 and tkinter with your package manager. On Ubuntu:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip python3-tk
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

Open a terminal in the project folder.

**macOS / Linux**

Replace the `cd` path with wherever you saved the folder:

```bash
cd ~/Desktop/slowphase-okr
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

You should see `(.venv)` at the start of your terminal prompt. That means the virtual environment is active.

**Windows (PowerShell)**

Replace the `cd` path with wherever you saved the folder:

```powershell
cd $env:USERPROFILE\Desktop\slowphase-okr
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

**Windows (Command Prompt)**

```cmd
cd %USERPROFILE%\Desktop\slowphase-okr
py -m venv .venv
.venv\Scripts\activate.bat
pip install -e .
```

If `py` is not found, use `python` instead in the commands above.

**What is the virtual environment?** It keeps this app’s packages separate from the rest of your system. You only need to activate it (the `source` or `Activate` step) each time you open a new terminal window before running the app.

### Step 4 — Run the app

With the virtual environment still active:

```bash
slowphase-okr
```

or

```bash
python -m slowphase_okr
```

On macOS/Linux you can also use `python3 -m slowphase_okr`.

A window should open. If nothing happens or you see an error, see [Troubleshooting](#troubleshooting).

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `python3` or `py` not found | Install Python from [python.org](https://www.python.org/downloads/). On Windows, reinstall with **Add to PATH** checked. |
| `pip` not found | Run `python3 -m pip install -e .` (macOS/Linux) or `py -m pip install -e .` (Windows) |
| PowerShell blocks activation | Use Command Prompt with `activate.bat`, or see the PowerShell note in Step 1 |
| `No module named '_tkinter'` (Linux) | Run `sudo apt install python3-tk` (Ubuntu) or install the tk package for your distro |
| App opens then closes immediately | Run `python -m slowphase_okr` from the terminal to see the error message |
| `slowphase-okr` command not found | Make sure the virtual environment is activated (`(.venv)` in your prompt), then try `python -m slowphase_okr` |

## Install (quick reference)

If Python is already set up and you have the repo:

**macOS / Linux**

```bash
cd slowphase-okr
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Windows**

```powershell
cd slowphase-okr
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

**Requirements:** Python 3.9+ and tkinter (included with the python.org installers on macOS and Windows).

## Run (quick reference)

With the virtual environment activated:

```bash
slowphase-okr
```

or

```bash
python -m slowphase_okr
```

On Windows, `python` is usually correct inside the venv. On macOS/Linux, `python3 -m slowphase_okr` also works.

## Workflow

1. **Browse gaze file** — your gaze direction file (see [Data format](#data-format))
2. **Browse time file** — your timestamp file (one time per gaze sample)
3. Enter **Trial ID** and **stimulus velocity** (default 31 deg/s, press Enter to apply after editing)
4. Click **Load trial**
5. **Click twice** on the elevation trace: start and end of each upward slow phase (snaps to nearest sample)
6. **Scroll** to zoom the time axis, **←/→** to pan, and use **2 s / 5 s / 10 s / Full / Reset** view buttons
7. Press **Accept segment** (`A`) to keep the fit
8. **Export Excel** when done

Hover over the plot to see time and elevation at the nearest sample. Press **`?`** (or the **?** button) for the full shortcut list.

## Features

| Feature | Description |
|---------|-------------|
| **Analysis window** | Full trial — first timestamp to last timestamp |
| **Segment list** | Panel showing #, times, gain, R², upward flag. Click to select. |
| **Segment labels** | `#N` badges on accepted segments in the plot |
| **Edit segments** | Delete selected (`Del`), undo last (`U`), nudge boundaries (`[ ]` start, `, .` end) |
| **Stimulus velocity** | Editable anytime. Enter recalculates gains for all segments. |
| **JSON autosave** | Saves to `{trial_id}_slowphase_okr_autosave.json` in the trial folder. Offers restore on reload. |
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

The GUI expects **two text files per trial**: one with gaze direction and one with timestamps. Filenames do not matter — use whatever your pipeline exports.

### Gaze + timestamp files (GUI)

This is the format the app loads.

| File | What it contains |
|------|------------------|
| **Gaze file** | One 3D gaze direction per line, as `(x, y, z)` tuples |
| **Time file** | One timestamp per line, in seconds, aligned sample-for-sample with the gaze file |

**Example filenames** (from our Vive Pro Eye / Tobii + Unity setup): `rotatedGaze.txt` and `gazeTime.txt`. Your lab may use different names — that is fine as long as the content matches.

This tool was developed for our lab to support gaze data recorded with a **HTC Vive Pro Eye** headset (Tobii eye tracking). The gaze file holds direction vectors in the experiment’s rotated coordinate frame. The app converts `(x, y, z)` to elevation (`azimuth = atan2(x,z)`, `elevation = asin(y/r)`).

**Requirements**

- The same number of lines in both files
- Timestamps in seconds
- Gaze lines parseable as `(x, y, z)` floats (see `examples/` for sample files)


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
