# Zero-To-CAD Local Tool

Desktop application for running [ADSKAILab/Zero-To-CAD-Qwen3-VL-2B](https://huggingface.co/ADSKAILab/Zero-To-CAD-Qwen3-VL-2B) inference locally, browsing the [test split](https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-1m/tree/main/data/test) of [Zero-To-CAD-1m](https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-1m), and visualizing input views plus generated CadQuery solids.

## Features

- **Lazy parquet browser** — index 10k test samples by UUID without unpacking images upfront
- **Per-row export** — write `view_0..7.png`, `code.py`, `model.step`, `model.stl` for inspection
- **Custom inference** — drag-and-drop or load 8 PNG views and generate CadQuery code
- **Sandboxed execution** — run generated code in an isolated subprocess; view STL in embedded VTK viewer
- **Side-by-side 3D** — predicted mesh vs ground-truth mesh from dataset
- **Drop test** — re-execute the current prediction as an N-copy grid assembly (`Drop test…` toolbar button)
- **Reasoning test** — with Cosmos-Reason loaded, ask it to explain a selected row from the ground-truth CadQuery code
- **Save asset** — bundle the latest prediction into a portable asset folder (STEP + STL + OBJ/MTL + textures + input views + code + manifest)

## Requirements

- Linux (tested on Ubuntu 24.04)
- NVIDIA GPU with CUDA 12.x for the local (vLLM) models
- ~8 GB disk for test parquet + model weights (Hugging Face cache)
- Python 3.12 (CadQuery wheels do not support 3.13+)
- An API key only if you use a hosted model: `ANTHROPIC_API_KEY` (Claude Fable) or `GEMINI_API_KEY` (Gemini). Set these in the in-app **Settings…** dialog (stored in a gitignored `settings.json`) or as environment variables.

## Install

```bash
cd Zero-To-CAD
conda env create -f environment.yml
conda activate zero-to-cad
```

Or install into an existing Python 3.12 environment:

```bash
pip install -e ".[dev]"
```

### vLLM (local inference)

Local models run on an **in-process vLLM engine** (no separate server). `vllm`
pulls a compatible `torch` build and is GPU/CUDA-specific, so install it into the
conda env after creating it:

```bash
conda activate zero-to-cad
pip install vllm
```

`environment.yml` and `pyproject.toml` already list `vllm`, but if it was skipped
(or you need a CUDA-specific build), the command above adds it to the env. Tuning
is env-overridable: `VLLM_MAX_MODEL_LEN` (default `8192`) and
`VLLM_GPU_MEMORY_UTILIZATION` (default `0.9`).

### Hosted models (API keys)

Claude Fable and Gemini are cloud API calls — no local weights. Provide the
relevant key in one of two ways:

- **Settings… dialog** (recommended): toolbar → *Settings…* → enter your key.
  Values are saved to `settings.json` at the repo root, which is gitignored so
  secrets are never committed. The file is created with `0600` permissions.
- **Environment variables**: `ANTHROPIC_API_KEY` (Claude) and `GEMINI_API_KEY`
  (Gemini). Environment variables take precedence over `settings.json`.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
```

Other endpoints (`COSMOS3_NANO_BASE_URL`, `VLLM_REMOTE_BASE_URL`) can also be
set from the Settings… dialog. Override the path/location with the
`ZERO_TO_CAD_SETTINGS` environment variable.

## Run

```bash
conda activate zero-to-cad
python -m zero_to_cad
# or
zero-to-cad
```

### First-run workflow

1. **Download test split** — left panel → *Download test split…* (~3.4 GB, 24 parquet files)
2. **Load model** — toolbar → pick a model from the combo box → *Load model* (downloads weights on first use)

### Switching models

The toolbar combo box lists every model in the hardcoded registry ([`src/zero_to_cad/config.py`](src/zero_to_cad/config.py)):

| Model | Backend | Notes |
|-------|---------|-------|
| **Zero-To-CAD 2B (CadQuery fine-tune)** | vLLM | CadQuery fine-tune; fits most CUDA GPUs |
| **Cosmos-Reason2 8B + CadQuery docs** | vLLM | Baseline with condensed CadQuery API reference in the system prompt; gated; ~32 GB GPU memory |
| **Cosmos3 8B (Zero-To-CAD reasoning fine-tune)** | vLLM | Local reasoning fine-tune; override path with `COSMOS3_MODEL` |
| **Cosmos3-Nano + CadQuery docs** | Remote vLLM (OpenAI API) | `cosmos3_omni` architecture that stock in-process vLLM can't load; served by a separate Cosmos3-capable vLLM server. Set `COSMOS3_NANO_BASE_URL`. |
| **Claude Fable 5** | Anthropic API | Cloud model; needs `ANTHROPIC_API_KEY`; override id with `CLAUDE_FABLE_MODEL` |
| **Gemini 2.5 Pro** | Google Gemini API | Cloud model; needs `GEMINI_API_KEY`; override id with `GEMINI_MODEL` |

Only one model is loaded at a time. Selecting a different entry and clicking *Load model* releases the previous weights from VRAM before loading the new one.

Gated baselines (Cosmos-Reason2-8B, Cosmos3-Nano) require Hugging Face authentication:

1. Accept the model gate (e.g. [nvidia/Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B))
2. Run `huggingface-cli login`

The Claude Fable and Gemini models are API calls — set `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` via the **Settings…** dialog (or your environment) before loading them.

**Cosmos3-Nano** uses NVIDIA's `cosmos3_omni` architecture, which stock in-process vLLM cannot load. Run it on a separate Cosmos3-capable vLLM server and point the app at it. For example:

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano \
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
  --allowed-local-media-path / --port 8000 --init-timeout 1800
```

Then set `COSMOS3_NANO_BASE_URL` (default `http://localhost:8000/v1`) before launching the app. "Load model" verifies the server is reachable.

Each model uses its own system/user prompt and `backend` (also defined in `config.MODELS`). To add another model, append a `ModelEntry` with its id, label, prompts, and backend (`vllm`, `openai`, `anthropic`, or `gemini`).

### Run history

Every *Generate* run is logged on the **History** tab (next to Predicted / Ground truth). Click an entry to restore its input views, generated code, and mesh (if you already executed it). Use this to compare outputs from different models on the same sample without re-running inference.

### First-run workflow (continued)

3. **Browse** — select a UUID to load 8 views and ground-truth code
4. **Generate** — toolbar → *Generate* (requires model + 8 views)
5. **Reasoning test** — if Cosmos-Reason is loaded, toolbar → *Reasoning test* to generate reasoning from the selected row's ground-truth code
6. **Execute** — toolbar → *Execute* to run predicted code and show the mesh
7. **Drop test** — toolbar → *Drop test…* to re-run the prediction as N copies arranged on a grid
8. **Save asset** — toolbar → *Save asset…* to write `<asset>/{model.step, model.stl, model.obj, model.mtl, textures/, views/, code.py, manifest.json}` for use in Blender / three.js / game engines
9. **Export** — toolbar → *Export row…* to write a human-readable folder for one dataset sample

### Custom images (no dataset)

1. Toolbar → *Load 8 PNGs…* or drag images onto each view tile
2. *Load model* → *Generate* → *Execute*

## Project layout

```
src/zero_to_cad/
  dataset/     # parquet download + lazy index
  inference/   # vLLM, Anthropic, Gemini, remote backends, factory, prompts
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

The vLLM backend sets `VLLM_USE_FLASHINFER_SAMPLER=0` automatically, since the flashinfer sampler's JIT arch probe fails on Blackwell (sm_120) GPUs; vLLM falls back to the native Torch sampler. Make sure your installed `vllm`/`torch` build targets your CUDA version.

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

### Local inference (vLLM)

Local models load through an in-process `vllm.LLM` engine with `trust_remote_code=True` (required for the Cosmos3-Nano `cosmos3_omni` architecture). Switching models releases the engine and frees GPU memory before the next one loads.

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
