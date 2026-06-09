# Run Qwen with vLLM on AMD Jupyter

## 1. Start the vLLM server on port 8000

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-7B-Instruct \
  --port 8000 \
  --host 0.0.0.0
```

Wait until the log prints:

```text
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Run in a separate terminal or append `> vllm.log 2>&1 &` to background it.

## 2. Test Qwen locally with curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-7B-Instruct",
    "messages": [{"role": "user", "content": "/no_think What is the root cause?"}],
    "max_tokens": 128,
    "temperature": 0.1
  }'
```

Expected: a JSON response with `choices[0].message.content` containing a short answer.

## 3. Start Streamlit with QWEN_BASE_URL

```bash
QWEN_BASE_URL=http://localhost:8000/v1 \
QWEN_MODEL=Qwen/Qwen2-7B-Instruct \
python -m streamlit run app/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false \
  > streamlit.log 2>&1 &
```

The Streamlit cockpit will show a green **Live LLM** badge in the Ask InfraGraph tab when `QWEN_BASE_URL` is set.

## 4. Open Streamlit through /proxy/8501/

In the Jupyter browser:

```text
https://<your-jupyter-host>/proxy/8501/
```

If the proxy gives a 502 or blank page, use localtunnel (section 5).

## 5. Localtunnel for Streamlit (if proxy fails)

```bash
npx localtunnel --port 8501
```

Open the printed URL, for example `https://xxxxx.loca.lt`.  
If a password is requested:

```bash
curl https://loca.lt/mytunnelpassword
```

## 6. Localtunnel for vLLM (call AMD-hosted Qwen from a local laptop)

If your laptop browser needs to reach the AMD-hosted Qwen API directly, tunnel vLLM too:

```bash
npx localtunnel --port 8000
# prints e.g. https://yyyyy.loca.lt
```

Set on your laptop:

```bash
export QWEN_BASE_URL=https://yyyyy.loca.lt/v1
export QWEN_MODEL=Qwen/Qwen2-7B-Instruct
streamlit run app/streamlit_app.py
```

The Streamlit app adds the `Bypass-Tunnel-Reminder: true` header automatically so the localtunnel interstitial page is skipped.

## 7. Windows curl example with Bypass-Tunnel-Reminder

From a Windows laptop (PowerShell or cmd) calling the vLLM localtunnel:

```powershell
curl.exe -X POST https://yyyyy.loca.lt/v1/chat/completions `
  -H "Content-Type: application/json" `
  -H "Bypass-Tunnel-Reminder: true" `
  -d "{\"model\":\"Qwen/Qwen2-7B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"/no_think ping\"}],\"max_tokens\":32}"
```

## 8. Troubleshooting dependency issues

### blinker conflict

```bash
python -m pip install --ignore-installed blinker streamlit pandas plotly requests altair pydeck gitpython --no-cache-dir
```

### starlette / protobuf / numpy version conflicts

```bash
python -m pip install "starlette<0.49.0" "protobuf<7.0.0" "numpy<2.3"
```

Run this after the initial streamlit install if you see import errors on startup.

### `streamlit: command not found`

```bash
python -m streamlit run app/streamlit_app.py
```

### vLLM CUDA / ROCm OOM

Reduce the model or set `--gpu-memory-utilization 0.85` in the vLLM launch command.

### LLM call fails in the chat tab

The cockpit automatically falls back to deterministic local answers and shows:

```text
⚠ Live LLM unreachable — showing local answer.
```

Check `vllm.log` and confirm `curl http://localhost:8000/v1/models` returns your model.
