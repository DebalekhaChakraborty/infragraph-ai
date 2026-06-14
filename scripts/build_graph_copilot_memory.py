"""Build the Graph Copilot global vector memory index.

Loads every enterprise graph artifact across all scenarios and indexes
fine-grained facts (nodes, edges, cross-diagram edges, IPs, alerts,
propagation steps, GNN candidates, impact-path edges) into the ChromaDB
collection ``infragraph_global_memory``.

Usage:
    python scripts/build_graph_copilot_memory.py [--scenario SCENARIO_ID]
    python scripts/build_graph_copilot_memory.py --dry-run

The script is safe to re-run; it upserts documents by their content-hash ID.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR   = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

GLOBAL_GRAPH_DIR = REPO_ROOT / "runtime_state" / "global_graph_memory"
SCENARIOS_DIR    = REPO_ROOT / "assets" / "scenarios"
RUNTIME_DIR      = REPO_ROOT / "runtime_state" / "live_ingestion"
COLLECTION_NAME  = "infragraph_global_memory"


def _safe_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_json_list(p: Path) -> list:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _collect_scenario_dirs() -> list[Path]:
    dirs: list[Path] = []
    for base in (SCENARIOS_DIR, RUNTIME_DIR):
        if base.exists():
            dirs.extend(sorted(p for p in base.iterdir() if p.is_dir()))
    return dirs


def build_all_docs(scenario_filter: str = "") -> list[dict]:
    from vector_memory.index_builder import build_vector_docs_from_enterprise_graph  # type: ignore

    all_docs: list[dict] = []
    seen_scenario_ids: set[str] = set()

    # ── Global graph summary ─────────────────────────────────────────────────
    global_graph_path = GLOBAL_GRAPH_DIR / "infragraph_global_graph.json"
    if global_graph_path.exists():
        print(f"Loading global graph: {global_graph_path}")
        global_graph = _safe_json(global_graph_path)
        global_scen_id = "global"
        docs = build_vector_docs_from_enterprise_graph(
            enterprise_graph=global_graph,
            scenario_id=global_scen_id,
        )
        all_docs.extend(docs)
        print(f"  → {len(docs)} docs from global graph")

    # ── Per-scenario enterprise graphs ───────────────────────────────────────
    for sdir in _collect_scenario_dirs():
        eg_path = sdir / "enterprise_graph.json"
        if not eg_path.exists():
            continue

        eg = _safe_json(eg_path)
        scenario_id = str(eg.get("scenario_id") or sdir.name)

        if scenario_filter and scenario_id != scenario_filter:
            continue
        if scenario_id in seen_scenario_ids:
            continue
        seen_scenario_ids.add(scenario_id)

        alerts = _safe_json_list(sdir / "alerts.json")
        incident_path = sdir / "enterprise_incident.json"
        if not incident_path.exists():
            incident_path = sdir / "incident.json"
        incident = _safe_json(incident_path) if incident_path.exists() else {}
        prop_steps = incident.get("propagation_steps") or []

        rca_path = sdir / "enterprise_gnn_rca_result.json"
        if not rca_path.exists():
            rca_path = sdir / "rca_result.json"
        rca = _safe_json(rca_path) if rca_path.exists() else {}

        alert_tl = incident.get("alert_timeline") or alerts

        docs = build_vector_docs_from_enterprise_graph(
            enterprise_graph=eg,
            scenario_id=scenario_id,
            alert_timeline=alert_tl,
            propagation_steps=prop_steps,
            enterprise_rca=rca,
        )
        all_docs.extend(docs)
        print(f"  scenario {scenario_id}: {len(docs)} docs")

    return all_docs


def upsert_to_chroma(docs: list[dict], dry_run: bool = False) -> tuple[int, str]:
    if dry_run:
        return len(docs), ""
    try:
        import chromadb  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        return 0, f"chromadb / sentence-transformers not installed: {exc}"

    try:
        client     = chromadb.PersistentClient(path=str(REPO_ROOT / "vector_store"))
        collection = client.get_or_create_collection(COLLECTION_NAME)
        model      = SentenceTransformer("all-MiniLM-L6-v2")

        batch_size = 200
        upserted   = 0
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            texts     = [d["text"]     for d in batch]
            ids       = [d["id"]       for d in batch]
            metadatas = [d["metadata"] for d in batch]
            embeddings = model.encode(texts, show_progress_bar=False).tolist()
            collection.upsert(ids=ids, embeddings=embeddings,
                              documents=texts, metadatas=metadatas)
            upserted += len(batch)

        return upserted, ""
    except Exception as exc:
        return 0, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Graph Copilot global memory index.")
    parser.add_argument("--scenario", default="", help="Index only this scenario ID.")
    parser.add_argument("--dry-run", action="store_true", help="Build docs but skip Chroma upsert.")
    args = parser.parse_args()

    print("Building Graph Copilot vector docs…")
    docs = build_all_docs(scenario_filter=args.scenario)

    counts: dict[str, int] = {}
    for d in docs:
        st = d.get("metadata", {}).get("source_type", "unknown")
        counts[st] = counts.get(st, 0) + 1

    print("\nDoc type breakdown:")
    for st, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {st:<35} {n}")

    nodes_indexed      = counts.get("enterprise_node", 0)
    edges_indexed      = counts.get("enterprise_edge", 0)
    cross_indexed      = counts.get("cross_diagram_edge", 0)
    alerts_indexed     = counts.get("alert_timeline_event", 0)
    rca_docs_indexed   = counts.get("gnn_candidate", 0) + counts.get("impact_path_edge", 0)
    total              = len(docs)

    print(f"\n  Nodes indexed:         {nodes_indexed}")
    print(f"  Edges indexed:         {edges_indexed}")
    print(f"  Cross-diagram indexed: {cross_indexed}")
    print(f"  Alerts indexed:        {alerts_indexed}")
    print(f"  RCA docs indexed:      {rca_docs_indexed}")
    print(f"  Total docs:            {total}")

    if args.dry_run:
        print("\n[dry-run] Skipping Chroma upsert.")
        return 0

    print(f"\nUpserting {total} docs into collection '{COLLECTION_NAME}'…")
    upserted, err = upsert_to_chroma(docs)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print(f"Done. {upserted} documents upserted into '{COLLECTION_NAME}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
