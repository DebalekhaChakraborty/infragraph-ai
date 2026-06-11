"""
prepare_verl_dataset.py — Convert InfraGraph RCA JSONL records to vERL parquet.

Reads:
    training/verl_grpo/data/rca_remediation_rl_train.jsonl
    training/verl_grpo/data/rca_remediation_rl_eval.jsonl

Writes:
    training/verl_grpo/data/verl_train.parquet
    training/verl_grpo/data/verl_eval.parquet

Each parquet row follows the vERL dataset schema:

    data_source   str     "infragraph_rca_remediation"
    prompt        str     JSON list of chat messages — [{"role":"user","content":...}]
    ability       str     "graph_grounded_remediation"
    reward_model  str     JSON containing style + ground_truth dict
    extra_info    str     JSON containing id, scenario_id, scope, root_cause, …
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
DATA_DIR   = SCRIPT_DIR / "data"

DEFAULT_TRAIN_JSONL = DATA_DIR / "rca_remediation_rl_train.jsonl"
DEFAULT_EVAL_JSONL  = DATA_DIR / "rca_remediation_rl_eval.jsonl"
DEFAULT_TRAIN_PARQ  = DATA_DIR / "verl_train.parquet"
DEFAULT_EVAL_PARQ   = DATA_DIR / "verl_eval.parquet"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"  [warn] skipping malformed line in {path.name}: {exc}",
                          file=sys.stderr)
    return records


def _build_prompt_messages(record: dict) -> str:
    """
    Extract the user-facing prompt content and format as a vERL chat messages list.

    vERL expects prompt as a JSON string containing a list of
    {"role": "user"|"assistant", "content": "..."} dicts.
    """
    raw_prompt: str = record.get("prompt", "")

    # The JSONL prompt field may already be a bare text block with
    # "[system]\n...\n[user]\n..." formatting produced by build_rca_rl_dataset.py.
    # Extract the [user] portion if present; otherwise use the full text.
    user_content = raw_prompt
    if "[user]" in raw_prompt.lower():
        parts = raw_prompt.split("\n")
        in_user = False
        user_lines: list[str] = []
        for line in parts:
            if line.strip().lower() in ("[user]", "user:", "[user]"):
                in_user = True
                continue
            if in_user and line.strip().startswith("[") and line.strip().endswith("]"):
                break
            if in_user:
                user_lines.append(line)
        user_content = "\n".join(user_lines).strip() or raw_prompt

    messages = [{"role": "user", "content": user_content}]
    return json.dumps(messages, ensure_ascii=False)


def _build_ground_truth(record: dict) -> str:
    """
    Build the ground_truth payload stored in reward_model.

    Contains all fields the reward function needs to score a response.
    """
    gt = {
        "root_cause":        record.get("root_cause", ""),
        "impacted_nodes":    record.get("impacted_nodes", []),
        "impacted_diagrams": record.get("impacted_diagrams", []),
        "graph_evidence":    record.get("graph_evidence", []),
        "scope":             record.get("scope", "enterprise"),
        "reward_tags":       record.get("reward_tags", []),
    }
    return json.dumps(gt, ensure_ascii=False)


def _build_reward_model(record: dict) -> str:
    rm = {
        "style":        "rule",
        "ground_truth": _build_ground_truth(record),
    }
    return json.dumps(rm, ensure_ascii=False)


def _build_extra_info(record: dict) -> str:
    ei = {
        "id":               record.get("id", ""),
        "scenario_id":      record.get("scenario_id", ""),
        "scope":            record.get("scope", "enterprise"),
        "root_cause":       record.get("root_cause", ""),
        "impacted_nodes":   record.get("impacted_nodes", []),
        "impacted_diagrams": record.get("impacted_diagrams", []),
    }
    return json.dumps(ei, ensure_ascii=False)


def _to_verl_row(record: dict) -> dict:
    return {
        "data_source":  "infragraph_rca_remediation",
        "prompt":       _build_prompt_messages(record),
        "ability":      "graph_grounded_remediation",
        "reward_model": _build_reward_model(record),
        "extra_info":   _build_extra_info(record),
    }


def _write_parquet(rows: list[dict], out_path: Path) -> None:
    """Write rows to parquet, preferring pandas; falls back to pyarrow directly."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"  {len(rows)} rows -> {out_path}  (via pandas)")
        return
    except ImportError:
        pass
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, str(out_path))
        print(f"  {len(rows)} rows -> {out_path}  (via pyarrow)")
        return
    except ImportError:
        pass
    raise RuntimeError(
        "Neither pandas nor pyarrow is installed. "
        "Install with:  pip install pandas pyarrow"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RCA JSONL records to vERL parquet format."
    )
    parser.add_argument("--train-jsonl", default=str(DEFAULT_TRAIN_JSONL),
                        help="Input train JSONL")
    parser.add_argument("--eval-jsonl",  default=str(DEFAULT_EVAL_JSONL),
                        help="Input eval JSONL")
    parser.add_argument("--train-parquet", default=str(DEFAULT_TRAIN_PARQ),
                        help="Output train parquet")
    parser.add_argument("--eval-parquet",  default=str(DEFAULT_EVAL_PARQ),
                        help="Output eval parquet")
    args = parser.parse_args()

    train_path = Path(args.train_jsonl)
    eval_path  = Path(args.eval_jsonl)

    if not train_path.exists():
        print(f"[ERROR] Train JSONL not found: {train_path}", file=sys.stderr)
        print("Run:  python training/verl_grpo/build_rca_rl_dataset.py", file=sys.stderr)
        sys.exit(1)
    if not eval_path.exists():
        print(f"[ERROR] Eval JSONL not found: {eval_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {train_path.name} ...")
    train_records = _read_jsonl(train_path)
    print(f"  {len(train_records)} train records")

    print(f"Reading {eval_path.name} ...")
    eval_records  = _read_jsonl(eval_path)
    print(f"  {len(eval_records)} eval records")

    train_rows = [_to_verl_row(r) for r in train_records]
    eval_rows  = [_to_verl_row(r) for r in eval_records]

    print("Writing parquet files ...")
    _write_parquet(train_rows, Path(args.train_parquet))
    _write_parquet(eval_rows,  Path(args.eval_parquet))
    print("Done.")


if __name__ == "__main__":
    main()
