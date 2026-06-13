"""
Chunk KB documents into retrievable passages.

chunk_document : split a single document dict into a list of chunk dicts
"""
from __future__ import annotations

import re


def chunk_document(
    doc: dict,
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[dict]:
    """
    Split a KB document body into chunks suitable for embedding.

    Strategy:
      1. Split on level-2 headings (``## ...``).  Each heading + its body
         becomes a candidate section.
      2. If a section exceeds ``max_chars``, split it further with
         character-level sliding windows using ``overlap`` chars of context.
      3. Prefix every chunk with the document title and section heading so
         the embedding captures the document context even in later chunks.

    Each returned dict:
      chunk_id    : ``{kb_id}::chunk-{i:03d}``   (0-based)
      evidence_id : ``KB-{kb_id}-{i:03d}``        (0-based)
      kb_id       : copied from doc
      title       : copied from doc
      doc_type    : copied from doc
      section     : heading of the section this chunk belongs to
      text        : chunk text (for embedding + prompt injection)
      metadata    : dict with all doc fields + chunk positioning info
    """
    kb_id    = doc["kb_id"]
    title    = doc["title"]
    body     = doc.get("body", "")

    sections = _split_into_sections(body)

    raw_chunks: list[tuple[str, str]] = []  # (section_heading, chunk_text)
    for heading, section_body in sections:
        # Always prefix with title and heading for embedding context
        prefix = f"[{title}] {heading}\n" if heading else f"[{title}]\n"
        full_section = prefix + section_body.strip()

        if len(full_section) <= max_chars:
            raw_chunks.append((heading, full_section))
        else:
            # Sliding window split
            for sub in _sliding_split(full_section, max_chars, overlap):
                raw_chunks.append((heading, sub))

    doc_type   = doc.get("doc_type", "")
    runbook_id = doc.get("runbook_id", "").strip()
    is_runbook = (doc_type == "runbook") and bool(runbook_id)

    result: list[dict] = []
    for i, (section, text) in enumerate(raw_chunks):
        chunk_id = f"{kb_id}::chunk-{i:03d}"
        if is_runbook:
            evidence_id = f"RB-{runbook_id}-{i:03d}"
        else:
            evidence_id = f"KB-{kb_id}-{i:03d}"

        meta: dict = {
            "chunk_id":               chunk_id,
            "evidence_id":            evidence_id,
            "kb_id":                  kb_id,
            "title":                  title,
            "doc_type":               doc_type,
            "section":                section,
            "owner_group":            doc.get("owner_group", ""),
            "applies_to_node_types":  doc.get("applies_to_node_types", []),
            "applies_to_diagrams":    doc.get("applies_to_diagrams", []),
            "applies_to_alert_types": doc.get("applies_to_alert_types", []),
            "rca_patterns":           doc.get("rca_patterns", []),
            "evidence_tags":          doc.get("evidence_tags", []),
            "source_type":            "sop_kb",
            "chunk_index":            i,
        }
        # Preserve runbook-specific fields in metadata for retriever / UI
        if is_runbook:
            meta["runbook_id"]          = runbook_id
            meta["source"]              = doc.get("source", "")
            meta["domain"]              = doc.get("domain", "")
            meta["approval_required"]   = doc.get("approval_required", False)
            meta["automation_eligible"] = doc.get("automation_eligible", False)
            meta["execution_mode"]      = doc.get("execution_mode", "")
            meta["tool_name"]           = doc.get("tool_name", "")
            meta["connector"]           = doc.get("connector", "")
            meta["action"]              = doc.get("action", "")
            meta["dry_run_supported"]   = doc.get("dry_run_supported", False)

        result.append({
            "chunk_id":    chunk_id,
            "evidence_id": evidence_id,
            "kb_id":       kb_id,
            "title":       title,
            "doc_type":    doc_type,
            "section":     section,
            "text":        text,
            "metadata":    meta,
        })

    return result


def _split_into_sections(body: str) -> list[tuple[str, str]]:
    """Return list of (heading, body) pairs split on '## ' level-2 headings."""
    # Split on lines that start with '## '
    pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    positions = [(m.start(), m.group(1).strip()) for m in pattern.finditer(body)]

    if not positions:
        return [("", body)]

    sections: list[tuple[str, str]] = []

    # Text before the first heading
    pre = body[: positions[0][0]].strip()
    if pre:
        sections.append(("", pre))

    for idx, (start, heading) in enumerate(positions):
        end = positions[idx + 1][0] if idx + 1 < len(positions) else len(body)
        # Skip past the heading line itself
        line_end = body.find("\n", start)
        section_body = body[line_end + 1 : end].strip() if line_end >= 0 else ""
        sections.append((heading, section_body))

    return sections


def _sliding_split(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split ``text`` into windows of at most ``max_chars`` with ``overlap``."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks
