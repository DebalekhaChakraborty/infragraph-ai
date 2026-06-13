"""
Constants and path defaults for the kb_retrieval package.
"""
from __future__ import annotations

import os

DEFAULT_KB_ROOT = "assets/kb"
DEFAULT_INDEX_DIR = "runtime_state/kb_index"
DEFAULT_COLLECTION = "infragraph_sop_kb"

# Override via INFRAGRAPH_EMBED_MODEL environment variable
DEFAULT_EMBED_MODEL = os.environ.get(
    "INFRAGRAPH_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# Document type to directory mapping (relative to kb_root)
DOC_TYPE_DIRS: dict[str, str] = {
    "sop":              "sops",
    "runbook":          "runbooks",
    "known_resolution": "known_resolutions",
}
