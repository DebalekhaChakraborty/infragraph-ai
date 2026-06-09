# Run InfraGraph AI Streamlit Cockpit in AMD Jupyter

## 1. Go to project folder

```bash
cd /workspace/shared/infragraph-ai
```

## 2. Install Streamlit dependencies

**Step 1 — install packages (handles the AMD blinker conflict):**

```bash
python -m pip install --ignore-installed blinker streamlit pandas plotly requests altair pydeck gitpython --no-cache-dir
```

The AMD Jupyter image may ship an old system-installed `blinker` package. `--ignore-installed blinker` skips the uninstall step and avoids the permission conflict that would otherwise abort the install.

**Step 2 — pin versions known to work on the AMD environment (team-tested):**

```bash
python -m pip install "starlette<0.49.0" "protobuf<7.0.0" "numpy<2.3"
```

Run step 2 after step 1 if Streamlit fails to start with import errors on `starlette`, `protobuf`, or `numpy`.

## 3. Start Streamlit

```bash
python -m streamlit run app/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false \
  > streamlit.log 2>&1 &
```

`--server.enableCORS false` and `--server.enableXsrfProtection false` are required when accessing Streamlit through the Jupyter proxy or a localtunnel URL, as both strip or modify request headers in ways that trigger Streamlit's CSRF protection.

## 4. Check Streamlit logs

```bash
tail -f streamlit.log
```

Expected output:

```text
Local URL: http://localhost:8501
Network URL: http://0.0.0.0:8501
```

Press `Ctrl + C` to stop watching logs. Streamlit continues running in the background.

## 5. Start localtunnel

Install localtunnel if needed:

```bash
sudo apt update && sudo apt install -y npm
npm install localtunnel
```

Expose Streamlit:

```bash
npx localtunnel --port 8501
```

Open the generated URL, for example:

```text
https://xxxxx.loca.lt
```

## 6. Localtunnel password

If localtunnel prompts for a password:

```bash
curl https://loca.lt/mytunnelpassword
```

Paste the printed value into the browser prompt.

## 7. Stop Streamlit

```bash
pkill -f streamlit
```

## 8. Restart cleanly

```bash
pkill -f streamlit

python -m streamlit run app/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false \
  > streamlit.log 2>&1 &

npx localtunnel --port 8501
```

## 9. Troubleshooting

### `streamlit: command not found`

```bash
python -m streamlit --version
python -m streamlit run app/streamlit_app.py
```

Always invoke via `python -m streamlit` rather than the bare `streamlit` binary.

### `Cannot uninstall blinker 1.4`

```bash
python -m pip install --ignore-installed blinker streamlit pandas plotly requests altair pydeck gitpython --no-cache-dir
```

### `/proxy/8501/` gives 502 Bad Gateway

The Jupyter proxy does not reliably forward WebSocket traffic. Use localtunnel instead:

```bash
npx localtunnel --port 8501
```

### Port already in use

```bash
pkill -f streamlit
```

Then restart Streamlit.

## 10. Expected MVP screen

The cockpit opens with a compact hero:

> *"Static diagram converted into graph memory. Alert stream analyzed using learned graph RCA."*

Use the sidebar **Workspace** radio to switch between two views:

**Workspace 1 — Diagram Intelligence**
- Input diagram vs YOLO detection (split view)
- Detected entity counts (7 device classes) + confidence table
- Extracted topology graph (NetworkX DiGraph — 17 nodes, 17 edges)
- Graph memory store: all 17 nodes with GNN scores + 16 extracted edges

**Workspace 2 — Alert RCA Intelligence**
- **Alert Investigation** — live alert stream, impacted nodes, topology image
- **GNN RCA Findings** — GNN winner card (FW-01 ✓), candidate ranking, model comparison, Model Evidence expander
- **GNN Propagation** — 5-step slider showing message-passing convergence from SW-CORE → FW-01
- **Operator Report** — Qwen-generated incident report with download button
- **Ask InfraGraph** — quick-question chips + free-text chat with deterministic fallback
