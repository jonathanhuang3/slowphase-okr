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

**What is the virtual environment?** It keeps this app’s packages separate from the rest of your system. You only run `pip install -e .` **once** (or again after pulling code updates). Closing the terminal does **not** remove the venv or your installed packages — you only need to activate it again before running the app.

### Step 3b — Auto-activate the virtual environment (optional)

If you do not want to run `source .venv/bin/activate` every time you open a new terminal, you can set up automatic activation.

**Important:** You do **not** need to run `pip install -e .` again when you open a new terminal. That step is one-time. Auto-activation only runs the activate step for you.

#### macOS / Linux (zsh — default on modern macOS)

Add this to `~/.zshrc` (open the file in a text editor, paste at the bottom, save):

```bash
# Auto-activate .venv when entering a project folder
_auto_venv() {
  if [[ -f .venv/bin/activate ]]; then
    if [[ "$VIRTUAL_ENV" != "$PWD/.venv" ]]; then
      source .venv/bin/activate
    fi
  elif [[ -n "$VIRTUAL_ENV" && "$VIRTUAL_ENV" != "$PWD/.venv" ]]; then
    deactivate 2>/dev/null
  fi
}
chpwd_functions+=(_auto_venv)
_auto_venv
```

Then reload your shell:

```bash
source ~/.zshrc
```

From now on, whenever you `cd` into `slowphase-okr` (or any folder that contains a `.venv`), the environment activates automatically and you should see `(.venv)` in your prompt.

#### macOS / Linux (bash)

Add the same block to `~/.bashrc` instead of `~/.zshrc`, but replace the last two lines with:

```bash
cd() {
  builtin cd "$@" || return
  if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
  fi
}
```

#### Windows (PowerShell)

Add this to your PowerShell profile (run `notepad $PROFILE` — create the file if prompted):

```powershell
function Enter-VenvIfPresent {
    if (Test-Path .venv\Scripts\Activate.ps1) {
        . .venv\Scripts\Activate.ps1
    }
}
function cd {
    param([string]$Path)
    if ($Path) { Set-Location $Path } else { Set-Location $HOME }
    Enter-VenvIfPresent
}
Enter-VenvIfPresent
```

Open a new PowerShell window after saving.

#### Alternative: direnv

If you use [direnv](https://direnv.net/), create a file named `.envrc` in the project root:

```bash
source .venv/bin/activate
```

Then run `direnv allow` once in that folder. direnv will activate the venv whenever you enter the directory.

### Step 4 — Run the app

#### First time (right after Step 3)

If you just finished installing, the virtual environment should already be active — you will see `(.venv)` at the start of your terminal line. Run:

```bash
slowphase-okr
```

A window should open.

#### Every time after that (new terminal window)

You only need **three steps**. You do **not** run `pip install` again.

1. Open **Terminal** (macOS/Linux) or **PowerShell** / **Command Prompt** (Windows).
2. Go to the project folder and **activate** the virtual environment.
3. Start the app.

**macOS / Linux** — replace the folder path with where yours lives:

```bash
cd ~/Desktop/slowphase-okr
source .venv/bin/activate
slowphase-okr
```

After step 2, your prompt should start with `(.venv)`. That means you are ready for step 3.

**Windows (PowerShell)**:

```powershell
cd $env:USERPROFILE\Desktop\slowphase-okr
.venv\Scripts\Activate.ps1
slowphase-okr
```

**Windows (Command Prompt)**:

```cmd
cd %USERPROFILE%\Desktop\slowphase-okr
.venv\Scripts\activate.bat
slowphase-okr
```

**If `slowphase-okr` does not work**, try this instead (with the venv still active):

```bash
python -m slowphase_okr
```

On macOS/Linux you can also use `python3 -m slowphase_okr`.

Closing the terminal or quitting the app does **not** uninstall anything. The next time, just repeat the three steps above. To skip the activate step automatically, see [Step 3b](#step-3b--auto-activate-the-virtual-environment-optional).

If nothing happens or you see an error, see [Troubleshooting](#troubleshooting).

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `python3` or `py` not found | Install Python from [python.org](https://www.python.org/downloads/). On Windows, reinstall with **Add to PATH** checked. |
| `pip` not found | Run `python3 -m pip install -e .` (macOS/Linux) or `py -m pip install -e .` (Windows) |
| PowerShell blocks activation | Use Command Prompt with `activate.bat`, or see the PowerShell note in Step 1 |
| `No module named '_tkinter'` (Linux) | Run `sudo apt install python3-tk` (Ubuntu) or install the tk package for your distro |
| App opens then closes immediately | Run `python -m slowphase_okr` from the terminal to see the error message |
| `slowphase-okr` command not found | Make sure the virtual environment is activated (`(.venv)` in your prompt), then try `python -m slowphase_okr` |
| Have to activate venv every new terminal | See [Step 3b](#step-3b--auto-activate-the-virtual-environment-optional) for auto-activation |
| Re-running `pip install` after closing terminal | Not needed — `pip install -e .` is one-time. Only activate the venv again (or set up auto-activation) |

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

`pip install -e .` is a one-time step. To skip manual activation in new terminals, see [Step 3b](#step-3b--auto-activate-the-virtual-environment-optional).

## Run (quick reference)

**Every time** you open a new terminal (after the one-time install in Step 3):

**macOS / Linux**

```bash
cd ~/Desktop/slowphase-okr          # 1. go to the project folder
source .venv/bin/activate           # 2. activate (look for (.venv) in your prompt)
slowphase-okr                       # 3. start the app
```

**Windows (PowerShell)**

```powershell
cd $env:USERPROFILE\Desktop\slowphase-okr
.venv\Scripts\Activate.ps1
slowphase-okr
```

Do **not** run `pip install -e .` again unless you updated the code or dependencies.

Alternative start command (if `slowphase-okr` is not found):

```bash
python -m slowphase_okr
```

On Windows, `python` is usually correct inside the venv. On macOS/Linux, `python3 -m slowphase_okr` also works.

## Workflow

1. **Browse gaze file** — your gaze direction file (see [Data format](#data-format))
2. **Browse time file** — your timestamp file (one time per gaze sample)
3. **Browse OKR log** *(optional)* — stimulus event log with block and fixation timing (see [OKR log](#okr-log-optional))
4. Enter **Trial ID** and **stimulus velocity** (default 31 deg/s, press Enter to apply after editing)
5. Click **Load trial**
6. **Click twice** on the elevation trace: start and end of each upward slow phase (snaps to nearest sample)
7. **Scroll vertically** or use **←/→** to pan the time axis (20% of window per step), and pick a **Window** (1 s, 2 s, 5 s, 10 s, or Full trial)
8. Press **Accept segment** (`A`) to keep the fit
9. **Export Excel** when done

Hover over the plot to see time and elevation at the nearest sample. Press **`?`** (or the **Help** button) for the full shortcut list.

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
| **OKR log markers** | Optional upload marks contrast-block starts (purple) and fixation-cross starts (gray) on the plot |
| **Invalid gaze samples** | `(NaN, NaN, NaN)` lines are kept for alignment but skipped for clicking and fitting |

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
| `←` / `→` | Pan view by 20% of visible window |
| Scroll (vertical) | Pan time forward / backward (20% of visible window per tick) |
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

**Example filenames** (from our Vive Pro Eye / Tobii + Unity setup): `rotatedGaze.txt` and `gazeTime.txt`, or raw SRanipal exports like `sranipalGazeSpace.txt` and `sranipalGazeTime.txt`. Your lab may use different names — that is fine as long as the content matches.

This tool was developed for our lab to support gaze data recorded with a **HTC Vive Pro Eye** headset (Tobii eye tracking). The gaze file holds direction vectors in the experiment’s rotated coordinate frame. The app converts `(x, y, z)` to elevation (`azimuth = atan2(x,z)`, `elevation = asin(y/r)`).

**Requirements**

- The same number of lines in both files (one timestamp per gaze line, including invalid samples)
- Timestamps in seconds
- Gaze lines as `(x, y, z)` tuples. Use `(NaN, NaN, NaN)` for missing tracking — these rows stay aligned with timestamps but are treated as invalid (no elevation, not clickable)

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
