# Evidence: AMD ROCm Qwen3-4B GRPO Training Run

This folder documents the completed AMD ROCm GRPO training pass for
InfraGraph AI's Qwen3-4B LoRA alignment pipeline.

## What completed

A real end-to-end GRPO training pass using `verl.trainer.main_ppo` with
`algorithm.adv_estimator=grpo` completed **32/32 training steps** on an
AMD ROCm GPU with Qwen/Qwen3-4B and LoRA rank 16.

## What is NOT claimed

- No reusable LoRA adapter checkpoint is committed to this repository.
- `pip install -r requirements.txt` cannot reproduce this training run.
- The base Streamlit demo does not require the AMD training stack.

## Files in this folder

| File | Contents |
|------|---------|
| `completion_evidence.md` | Full run configuration, patches applied, honest checkpoint status |
| `training_summary.md` | Auto-generated summary from `write_training_summary.py` |
| `torch_runtime.txt` | `torch.__version__`, HIP version, device info |
| `python_version.txt` | Python version used |
| `pip_freeze_successful_run.txt` | Full `pip freeze` output from the AMD training environment |
| `artifact_manifest.txt` | Paths to run artifacts (training_summary.md) |

## Reproducing the run

```bash
# On an AMD ROCm machine (not required for demo/app use):
bash scripts/amd_rocm/bootstrap_grpo_env.sh
bash scripts/amd_rocm/patch_verl_runtime_for_rocm.sh
INFRAGRAPH_RUN_REAL_VERL=1 bash training/verl_grpo/train_qwen3_grpo.sh
```

See [completion_evidence.md](completion_evidence.md) for the full list of
training overrides and runtime patches that were required.
