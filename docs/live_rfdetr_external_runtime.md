# Live RF-DETR External Runtime

InfraGraph AI keeps the UI, vision detector, and language model runtimes separate.

- Streamlit runs the application UI and session state.
- RF-DETR runs in an external detector Python process.
- Qwen/vLLM runs as a separate HTTP inference server.

This mirrors the existing Qwen/vLLM setup: Streamlit does not need the model package installed locally. The UI calls the detector runtime through a subprocess bridge and reads a structured JSON result.

## Why This Exists

The previous live ingestion path tried to import RF-DETR inside the Streamlit process. If the Streamlit virtual environment did not include `rfdetr`, onboarding reported `No module named 'rfdetr'` even when a base detector environment had the package installed.

The new path does not import RF-DETR in Streamlit. It runs:

```bash
python scripts/run_rfdetr_inference.py \
  --image /path/to/input.png \
  --checkpoint /path/to/checkpoint_best_total.pth \
  --out-json /tmp/infragraph_rfdetr_result.json \
  --out-image /tmp/infragraph_rfdetr_overlay.png \
  --confidence 0.25
```

## Environment Variables

```bash
INFRAGRAPH_RFDETR_PYTHON=/path/to/python
INFRAGRAPH_RFDETR_CHECKPOINT=/path/to/checkpoint_best_total.pth
INFRAGRAPH_RFDETR_CONFIDENCE=0.25
INFRAGRAPH_RFDETR_TIMEOUT=180
```

`INFRAGRAPH_RFDETR_PYTHON` defaults to the current Python after checking common detector locations such as `/opt/conda/bin/python`, `/usr/bin/python`, and `/workspace/shared/venvs/rfdetr/bin/python`.

`INFRAGRAPH_RFDETR_CHECKPOINT` can be set explicitly. If it is not set, the bridge searches common repository locations under `model_artifacts/` and `outputs/`.

## Detector Runtime Setup

Install detector dependencies in the detector or base environment, not necessarily the Streamlit environment:

```bash
pip install -r requirements/requirements-rfdetr-runtime.txt
```

Check the detector runtime:

```bash
python scripts/check_rfdetr_runtime.py --python /opt/conda/bin/python
```

Optionally run an image-level check:

```bash
python scripts/check_rfdetr_runtime.py \
  --python /opt/conda/bin/python \
  --image datasets/infragraph_v3/scenarios/train/enterprise_v3_0000/diagrams/branch_topology.png
```

## Streamlit Runtime Setup

Start Streamlit from the application environment and point it at the detector runtime:

```bash
source /workspace/shared/venvs/infragraph-app/bin/activate
export INFRAGRAPH_RFDETR_PYTHON=/opt/conda/bin/python
python -m streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

On Windows PowerShell:

```powershell
$env:INFRAGRAPH_RFDETR_PYTHON="D:\path\to\detector\python.exe"
python -m streamlit run app\streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

## UI Modes

The Diagram Intelligence UI distinguishes the active path explicitly:

- `LIVE_RFDETR_INFERENCE`: RF-DETR subprocess completed successfully and returned detections.
- `VERIFIED_ANNOTATION_FALLBACK`: live detector inference was unavailable, and verified curated annotation metadata is being used instead.
- `SESSION_MEMORY_ABSORPTION`: the generated graph memory is absorbed into the current Streamlit session and resets on refresh.

The UI must not present verified annotation output as live detector inference. When RF-DETR is unavailable, the exact subprocess error is shown and the verified path is labeled clearly.

## Qwen/vLLM

This change does not alter Qwen/vLLM configuration. Qwen remains an HTTP runtime configured with:

```bash
INFRAGRAPH_QWEN_BASE_URL=http://127.0.0.1:8000/v1
INFRAGRAPH_QWEN_MODEL=infragraph
```

