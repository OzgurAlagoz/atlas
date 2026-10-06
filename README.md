# ATLAS

**Semantic 3D Mapping and Vision-Language Navigation System**

ATLAS is a long-term project that aims to build a system which reconstructs a 3D map of
an indoor space from RGB/RGB-D video, remembers objects across viewpoints, answers
natural-language spatial queries, and turns them into navigation goals for a simulated robot.

This repository is being built step by step, from first principles. The current stage is the
video and visualization layer: reading frames from a source, drawing runtime statistics on
them, displaying them, and optionally recording the processed output.

## Current features

- Read frames from a video file or camera (`VideoSource`)
- Resize frames to a configured resolution
- Overlay source FPS, processing FPS and total frame count (`Renderer`)
- Optionally record the annotated output to a timestamped MP4 (`Recorder`)
- Command line interface for source, resolution and recording

Detection, tracking, geometry and mapping modules exist as placeholders and are not
implemented yet.

## Requirements

- Python 3.10+
- See `requirements.txt` (OpenCV, NumPy, pytest)

## Installation

```bash
git clone https://github.com/OzgurAlagoz/atlas.git
cd atlas

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # Linux / macOS

pip install -r requirements.txt
```

## Usage

The package lives under `src/`, so that directory has to be importable. The simplest way is
to set `PYTHONPATH` before running:

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "src;."
python -m atlas.main

# Linux / macOS
PYTHONPATH=src:. python -m atlas.main
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | value from `configs/config.py` | Path to a video file, or a camera index |
| `--res W H` | `780 540` | Width and height the frames are resized to |
| `--save` | off | Record the annotated output to `outputs/<timestamp>.mp4` |

Example:

```bash
python -m atlas.main --source datasets/videos/room.mp4 --res 640 480 --save
```

Press `q` in the video window to stop.

## Tests

```bash
pytest
```

`pyproject.toml` already tells pytest where the packages live, so no extra setup is needed.

## Project structure

```
atlas/
|-- configs/            # configuration values (paths, resolution)
|-- datasets/           # input videos (not tracked by git)
|-- outputs/            # recorded videos (not tracked by git)
|-- src/atlas/
|   |-- main.py         # entry point, argument parsing, main loop
|   |-- video/
|   |   |-- source.py   # VideoSource: open / read / close
|   |   `-- recorder.py # Recorder: VideoWriter wrapper
|   |-- visualization/
|   |   `-- renderer.py # Renderer: overlay and display
|   |-- detection/      # placeholder
|   |-- tracking/       # placeholder
|   `-- common/         # placeholder
|-- tests/              # unit tests
`-- pyproject.toml      # pytest configuration
```

## Notes

`datasets/` and `outputs/` are excluded from version control, so a fresh clone contains
neither input videos nor recordings. Put your own video under `datasets/videos/` and point
`configs/config.py` (or `--source`) at it before running.
