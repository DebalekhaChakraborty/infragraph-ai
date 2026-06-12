"""
prepare_verl_dataset.py — Convert InfraGraph RCA JSONL records to vERL parquet.

Reads:
    training/verl_grpo/data/rca_remediation_rl_train.jsonl
    training/verl_grpo/data/rca_remediation_rl_eval.jsonl

Writes:
    training/verl_grpo/data/verl_train.parquet
    training/verl_grpo/data/verl_eval.parquet

Each parquet row follows the vERL dataset schema:

    data_source   str                  "infragraph_rca_remediation"
    prompt        str                  JSON list of chat messages — [{"role":"user","content":...}]
    ability       str                  "graph_grounded_remediation"
    reward_model  struct/dict          nested: {style: str, ground_truth: struct}
    extra_info    struct/dict          nested: {index: int, id, scenario_id, scope, root_cause, …}

reward_model and extra_info are written as nested struct columns (not JSON strings)
so that vERL's rl_dataset.py can call row_dict["extra_info"].get("index", 0) directly.
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


def _build_ground_truth(record: dict) -> dict:
    """
    Build the ground_truth payload stored inside reward_model.

    Returned as a dict so parquet preserves it as a nested struct, not a string.
    """
    return {
        "root_cause":        record.get("root_cause", ""),
        "impacted_nodes":    record.get("impacted_nodes", []),
        "impacted_diagrams": record.get("impacted_diagrams", []),
        "graph_evidence":    record.get("graph_evidence", []),
        "scope":             record.get("scope", "enterprise"),
        "reward_tags":       record.get("reward_tags", []),
    }


def _build_reward_model(record: dict) -> dict:
    """Return reward_model as a dict so parquet stores it as a struct, not a string."""
    return {
        "style":        "rule",
        "ground_truth": _build_ground_truth(record),
    }


def _build_extra_info(record: dict, idx: int) -> dict:
    """
    Return extra_info as a dict with an explicit 'index' key.

    vERL's rl_dataset.py calls row_dict.get('extra_info', {}).get('index', 0),
    so extra_info must be a dict (not a JSON string) and must contain 'index'.
    """
    return {
        "index":           idx,
        "id":              record.get("id", ""),
        "scenario_id":     record.get("scenario_id", ""),
        "scope":           record.get("scope", "enterprise"),
        "root_cause":      record.get("root_cause", ""),
        "impacted_nodes":  record.get("impacted_nodes", []),
        "impacted_diagrams": record.get("impacted_diagrams", []),
    }


def _to_verl_row(record: dict, idx: int) -> dict:
    return {
        "data_source":  "infragraph_rca_remediation",
        "prompt":       _build_prompt_messages(record),
        "ability":      "graph_grounded_remediation",
        "reward_model": _build_reward_model(record),
        "extra_info":   _build_extra_info(record, idx),
    }


def _write_parquet(rows: list[dict], out_path: Path) -> None:
    """
    Write rows to parquet using datasets.Dataset so nested dicts (extra_info,
    reward_model) are stored as struct columns, not JSON strings.
    vERL's rl_dataset.py calls row_dict.get('extra_info', {}).get('index', 0)
    directly on the deserialised row — it must be a dict, not a string.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import Dataset
        ds = Dataset.from_list(rows)
        ds.to_parquet(str(out_path))
        print(f"  {len(rows)} rows -> {out_path}  (via datasets)")

        # Validation: confirm extra_info is a dict in the written data
        first_ei = rows[0]["extra_info"]
        print(f"  type(extra_info[0]) = {type(first_ei).__name__}  "
              f"index={first_ei.get('index')}  scope={first_ei.get('scope')}")
        return
    except ImportError:
        pass
    raise RuntimeError(
        "The 'datasets' package is required for parquet serialisation. "
        "Install with:  pip install datasets"
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

    train_rows = [_to_verl_row(r, i) for i, r in enumerate(train_records)]
    eval_rows  = [_to_verl_row(r, i) for i, r in enumerate(eval_records)]

    print("Writing parquet files ...")
    _write_parquet(train_rows, Path(args.train_parquet))
    _write_parquet(eval_rows,  Path(args.eval_parquet))
    print("Done.")


if __name__ == "__main__":
    main()
