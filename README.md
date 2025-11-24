# Hieroglyph Demo (EACL 2026)

This repository contains the official demo prototype accompanying the submission to **EACL 2026**.
It provides a real-time system for **hieroglyph classification and translation** using a live camera feed.

## Requirements

- Python 3.11 (tested with 3.11.9)
- Camera device with system-level access
- Internet connection on first run only (~2 GB model download)

---
## Installation

It is highly recommended to install the demo inside a virtual environment to avoid dependency conflicts.

### 1. Create & activate a virtual environment

**Windows**
```
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install the Hieroglyph Demo package
Once the virtual environment is activated, install the package via pip:

```
pip install git+https://github.com/said702/Hieroglyph-demo-EACL2026
```
ℹ️ **Note:** This command installs the demo package and all required dependencies inside the virtual environment you activated in step 1.

---

## Running the Demo

Start the application using:

    python -m hieroglyph_demo.main --camera <index>
    
### Which camera index should you use?

- If your laptop has an internal webcam AND you plug in a USB/document camera:
    → Use --camera 1

- If you only use one external camera (no internal webcam active):
    → Use --camera 0

⚠️ Note: The demo requires direct system-level access to the camera.
It does not run in isolated environments such as WSL, Docker, or sandboxed VMs, because these environments restrict hardware access.

------------------------------------------------------------

## Automatic Downloads (First Run Only)

The first time you run the demo, it will automatically download:

- Detectron2 repository (for segmentation)
- YOLO11 models
- HuggingFace translation model (~1.9 GB)

These downloads happen **only once** and will be re-used afterward.

---
### Keyboard Controls

While the prototype is running and the camera window is active, you can control the system with the following keys:

| Key        | Action |
|------------|--------|
| ENTER      | Start Hieroglyph Classification |
| t          | Enable/Disable Hieroglyph Translation (toggle) |
| x          | Increase feedback font size |
| y          | Decrease feedback font size |
| n          | Reset the prototype (clear internal state) |
| h          | Show/Hide Help |
| q          | Quit the program |

⚠️ Note: If keyboard shortcuts do not respond, click inside the camera window to give it focus.




