# Zero-To-CAD Local Tool

Desktop application for running [ADSKAILab/Zero-To-CAD-Qwen3-VL-2B](https://huggingface.co/ADSKAILab/Zero-To-CAD-Qwen3-VL-2B) inference locally, browsing the [test split](https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-1m/tree/main/data/test) of [Zero-To-CAD-1m](https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-1m), and visualizing input views plus generated CadQuery solids.

## Features

- **Lazy parquet browser** — index 10k test samples by UUID without unpacking images upfront
- **Per-row export** — write `view_0..7.png`, `code.py`, `model.step`, `model.stl` for inspection
- **Custom inference** — drag-and-drop or load 8 PNG views and generate CadQuery code
- **Sandboxed execution** — run generated code in an isolated subprocess; view STL in embedded VTK viewer
- **Side-by-side 3D** — predicted mesh vs ground-truth mesh from dataset

## Requirements

- Linux (tested on Ubuntu 24.04)
- NVIDIA GPU with CUDA 12.x (RTX 5090 / Blackwell needs PyTorch `cu128` wheels)
- ~8 GB disk for test parquet + ~4 GB for model weights (Hugging Face cache)
- Python 3.12 (CadQuery wheels do not support 3.13+)

## Install

```bash
cd Zero-To-CAD
conda env create -f environment.yml
conda activate zero-to-cad
```

Or install into an existing Python 3.12 environment:

```bash
pip install -e ".[dev]"
# PyTorch with CUDA 12.8 (Blackwell / RTX 50-series):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Run

```bash
conda activate zero-to-cad
python -m zero_to_cad
# or
zero-to-cad
```

### First-run workflow

1. **Download test split** — left panel → *Download test split…* (~3.4 GB, 24 parquet files)
2. **Load model** — toolbar → *Load model* (downloads ~4 GB on first use)
3. **Browse** — select a UUID to load 8 views and ground-truth code
4. **Generate** — toolbar → *Generate* (requires model + 8 views)
5. **Execute** — toolbar → *Execute* to run predicted code and show the mesh
6. **Export** — toolbar → *Export row…* to write a human-readable folder for one sample

### Custom images (no dataset)

1. Toolbar → *Load 8 PNGs…* or drag images onto each view tile
2. *Load model* → *Generate* → *Execute*

## Project layout

```
src/zero_to_cad/
  dataset/     # parquet download + lazy index
  inference/   # Qwen3-VL wrapper
  execute/     # CadQuery subprocess sandbox
  ui/          # PySide6 GUI
tests/
```

## Tests

```bash
pytest tests/ -v
```

CadQuery execution test is skipped if `cadquery` is not importable. Parquet tests always run.

## Troubleshooting

### RTX 5090 / Blackwell (sm_120)

Use PyTorch built for CUDA 12.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3D viewports (desktop artifacts / blank panels)

By default the app uses an **image-based viewer** (`ZERO_TO_CAD_VIEWER=image`): meshes are rendered offscreen with PyVista and shown in the right panels. **Drag** to orbit, **scroll** to zoom, toggle **Wireframe edges**, or **Reset view**. This avoids a known Linux + Qt6 + VTK issue where embedded OpenGL widgets show stale desktop pixels instead of the scene.

For an **interactive** VTK widget (rotate/zoom with the mouse):

```bash
ZERO_TO_CAD_VIEWER=vtk python -m zero_to_cad
```

If the VTK backend still shows artifacts:

- Run on a machine with a working display (`echo $DISPLAY` should be set).
- Ensure `pyvistaqt`, `vtk`, and `pyside6` are from the same `zero-to-cad` conda env.
- On Linux over SSH, use `ssh -X` or run on the local workstation console.
- STEP meshes are tessellated through CadQuery automatically.

### Transformers version

This project pins `transformers>=4.57,<5`. The model card notes that `image-to-text` pipelines changed in v5; we load `Qwen3VLForConditionalGeneration` directly.

### Generated code imports

The sandbox only provides `cadquery` / `cq` in the execution namespace. Code that imports other packages will fail until the runner namespace is extended.

### Hugging Face authentication

Public model and dataset repos should work without a token. For rate limits, run `huggingface-cli login`.

## Screenshot

<!-- Replace with an actual screenshot after first run -->
_Placeholder: run the app and capture the main window with dataset browser, input views, code panel, and 3D viewers._

## License

Apache-2.0 (model and dataset are also Apache-2.0).

## References

- [Zero-To-CAD model](https://huggingface.co/ADSKAILab/Zero-To-CAD-Qwen3-VL-2B)
- [Zero-To-CAD-1m dataset](https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-1m)
- [CadQuery](https://github.com/CadQuery/cadquery)
- [Paper (arXiv:2604.24479)](https://arxiv.org/abs/2604.24479)
