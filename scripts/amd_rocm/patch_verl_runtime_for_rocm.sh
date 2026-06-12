#!/usr/bin/env bash
set -euo pipefail

echo "Patching vERL/vLLM runtime for AMD ROCm compatibility..."

python - <<'PY'
from pathlib import Path

# Patch 1: disable vLLM sleep mode in vERL async server
p = Path("/usr/local/lib/python3.12/dist-packages/verl/workers/rollout/vllm_rollout/vllm_async_server.py")
if p.exists():
    text = p.read_text()
    p.with_suffix(".py.bak_infragraph_rocm").write_text(text)

    text = text.replace(
        'logger.info(f"enable_sleep_mode: {self.config.enable_sleep_mode}")',
        'logger.info("enable_sleep_mode: False  # forced for ROCm")'
    )
    text = text.replace(
        '"enable_sleep_mode": self.config.enable_sleep_mode,',
        '"enable_sleep_mode": False,  # InfraGraph ROCm patch: unsupported on current platform'
    )
    p.write_text(text)
    print("Patched:", p)
else:
    print("Missing:", p)

# Patch 2: tolerate string extra_info/reward_model in vERL dataset reader
p = Path("/usr/local/lib/python3.12/dist-packages/verl/utils/dataset/rl_dataset.py")
if p.exists():
    text = p.read_text()
    p.with_suffix(".py.bak_infragraph_extra_info").write_text(text)

    old = '        index = row_dict.get("extra_info", {}).get("index", 0)\n'
    new = '''        # InfraGraph patch: tolerate JSON-string extra_info / reward_model.
        extra_info = row_dict.get("extra_info", {})
        if isinstance(extra_info, str):
            import json
            try:
                extra_info = json.loads(extra_info)
            except Exception:
                extra_info = {"raw_extra_info": extra_info}
        if not isinstance(extra_info, dict):
            extra_info = {"raw_extra_info": str(extra_info)}
        row_dict["extra_info"] = extra_info

        reward_model = row_dict.get("reward_model", {})
        if isinstance(reward_model, str):
            import json
            try:
                reward_model = json.loads(reward_model)
            except Exception:
                reward_model = {"raw_reward_model": reward_model}
        row_dict["reward_model"] = reward_model

        index = extra_info.get("index", 0)
'''
    if old in text and "InfraGraph patch: tolerate JSON-string extra_info" not in text:
        text = text.replace(old, new, 1)
        p.write_text(text)
        print("Patched:", p)
    else:
        print("Dataset patch already present or anchor not found:", p)
else:
    print("Missing:", p)
PY

echo "Runtime patch complete."
