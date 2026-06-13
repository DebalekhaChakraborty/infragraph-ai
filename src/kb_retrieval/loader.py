"""
KB document loader.

parse_frontmatter : parse YAML-like frontmatter (between --- delimiters)
load_kb_documents : load all markdown documents from the KB root directory
"""
from __future__ import annotations

from pathlib import Path


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Parse YAML-like frontmatter from the beginning of a document.

    Frontmatter must be enclosed between ``---`` lines at the very start
    of the file.  Returns (metadata_dict, body_text).

    Supports:
      - scalar values:  key: value
      - quoted strings: key: "value with spaces"
      - block lists:    key:\\n  - item1\\n  - item2
    """
    text = text.lstrip("﻿")  # strip BOM
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, stripped

    end = stripped.find("\n---", 3)
    if end < 0:
        return {}, stripped

    fm_text = stripped[3:end].strip()
    body = stripped[end + 4:].lstrip("\n")

    meta: dict = {}
    current_key: str | None = None
    current_list: list | None = None

    for line in fm_text.splitlines():
        # Block list item
        if line.startswith("  - ") or line.startswith("- "):
            val = line.lstrip(" -").strip().strip('"').strip("'")
            if current_key is not None and isinstance(meta.get(current_key), list):
                meta[current_key].append(val)
            continue

        if ":" in line:
            key_part, _, val_part = line.partition(":")
            key = key_part.strip()
            val = val_part.strip().strip('"').strip("'")
            current_key = key

            if not val:
                # Next lines will be list items
                meta[key] = []
                current_list = meta[key]
            elif val.startswith("[") and val.endswith("]"):
                # Inline list: [a, b, c]
                inner = val[1:-1]
                meta[key] = [
                    v.strip().strip('"').strip("'")
                    for v in inner.split(",")
                    if v.strip()
                ]
                current_list = None
            else:
                meta[key] = val
                current_list = None
        else:
            current_list = None

    return meta, body


def load_kb_documents(kb_root: Path) -> list[dict]:
    """
    Recursively load all markdown files from ``kb_root``.

    Each returned dict contains:
      kb_id        : from frontmatter or derived from filename
      title        : from frontmatter or filename
      doc_type     : from frontmatter or inferred from parent directory name
      version      : from frontmatter (str) or "unknown"
      owner_group  : from frontmatter or ""
      applies_to_node_types  : list[str]
      applies_to_diagrams    : list[str]
      applies_to_alert_types : list[str]
      rca_patterns           : list[str]
      evidence_tags          : list[str]
      file_path    : absolute path (str)
      body         : document body text (str)
      raw          : full raw file text (str)
    """
    documents: list[dict] = []

    for md_file in sorted(kb_root.rglob("*.md")):
        raw = md_file.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)

        # Infer doc_type from parent directory name if not in frontmatter
        parent_name = md_file.parent.name.lower()
        _dir_to_type = {
            "sops": "sop",
            "runbooks": "runbook",
            "known_resolutions": "known_resolution",
        }
        inferred_doc_type = _dir_to_type.get(parent_name, parent_name)

        # Derive kb_id from filename if not in frontmatter
        kb_id = str(meta.get("kb_id", "")).strip() or md_file.stem

        doc_type_str = str(meta.get("doc_type", inferred_doc_type)).strip()

        documents.append({
            "kb_id":        kb_id,
            "title":        str(meta.get("title", md_file.stem)).strip(),
            "doc_type":     doc_type_str,
            "version":      str(meta.get("version", "unknown")).strip(),
            "owner_group":  str(meta.get("owner_group", "")).strip(),
            "applies_to_node_types":  _as_list(meta.get("applies_to_node_types")),
            "applies_to_diagrams":    _as_list(meta.get("applies_to_diagrams")),
            "applies_to_alert_types": _as_list(meta.get("applies_to_alert_types")),
            "rca_patterns":           _as_list(meta.get("rca_patterns")),
            "evidence_tags":          _as_list(meta.get("evidence_tags")),
            # Runbook-specific fields (empty string / False when not a runbook)
            "runbook_id":          str(meta.get("runbook_id", "")).strip(),
            "source":              str(meta.get("source", "")).strip(),
            "domain":              str(meta.get("domain", "")).strip(),
            "approval_required":   _as_bool(meta.get("approval_required")),
            "automation_eligible": _as_bool(meta.get("automation_eligible")),
            "execution_mode":      str(meta.get("execution_mode", "")).strip(),
            "tool_name":           str(meta.get("tool_name", "")).strip(),
            "connector":           str(meta.get("connector", "")).strip(),
            "action":              str(meta.get("action", "")).strip(),
            "dry_run_supported":   _as_bool(meta.get("dry_run_supported")),
            "file_path":    str(md_file),
            "body":         body,
            "raw":          raw,
        })

    return documents


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1")
    return bool(value)
