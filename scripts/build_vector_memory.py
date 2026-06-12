#!/usr/bin/env python3
"""Build a local Chroma vector memory index from InfraGraph evidence files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _add_src(repo_root: Path) -> None:
    src = str(repo_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _docs_from_live_ingestion(repo_root: Path, build_docs) -> list[dict]:
    docs: list[dict] = []
    root = repo_root / "outputs" / "live_ingestion"
    if not root.exists():
        return docs
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        packet = _load_json(run_dir / "graph_memory_packet.json")
        local_graph = _load_json(run_dir / "local_graph.json")
        if not local_graph:
            local_graph = {"nodes": packet.get("nodes", []), "edges": packet.get("edges", [])}
        docs.extend(build_docs(packet, local_graph=local_graph))
    return docs


def _docs_from_incident_runs(repo_root: Path, build_docs) -> list[dict]:
    docs: list[dict] = []
    root = repo_root / "outputs" / "incident_runs"
    if not root.exists():
        return docs
    for run_dir in sorted(p for p in root.rglob("*") if p.is_dir()):
        local_incident = _load_json(run_dir / "local_incident.json")
        enterprise_incident = _load_json(run_dir / "enterprise_incident.json")
        local_rca = _load_json(run_dir / "local_rca_result.json")
        enterprise_rca = _load_json(run_dir / "enterprise_rca_result.json")
        if local_incident or enterprise_incident or local_rca or enterprise_rca:
            docs.extend(build_docs(
                {"run_id": run_dir.name},
                local_incident=local_incident,
                enterprise_incident=enterprise_incident,
                local_rca_result=local_rca,
                enterprise_rca_result=enterprise_rca,
            ))
    return docs


def _docs_from_enterprise_gnn(repo_root: Path, build_docs) -> list[dict]:
    docs: list[dict] = []
    root = repo_root / "outputs" / "enterprise_gnn_rca"
    if not root.exists():
        return docs
    for result_path in sorted(root.glob("*_enterprise_gnn_rca_result.json")):
        rca = _load_json(result_path)
        docs.extend(build_docs(
            {
                "scenario_id": rca.get("scenario_id", ""),
                "run_id": result_path.stem,
            },
            enterprise_rca_result=rca,
        ))
    return docs


def _docs_from_v3_scenarios(repo_root: Path, build_docs) -> list[dict]:
    docs: list[dict] = []
    root = repo_root / "datasets" / "infragraph_v3" / "scenarios"
    if not root.exists():
        return docs
    for scenario_dir in sorted(root.glob("*/*")):
        if not scenario_dir.is_dir():
            continue
        enterprise_graph = _load_json(scenario_dir / "enterprise_graph.json")
        alerts = _load_json(scenario_dir / "alerts.json")
        if not enterprise_graph and not alerts:
            continue
        packet = {
            "scenario_id": enterprise_graph.get("scenario_id") or alerts.get("scenario_id") or scenario_dir.name,
            "run_id": scenario_dir.name,
        }
        docs.extend(build_docs(
            packet,
            enterprise_graph=enterprise_graph,
            enterprise_incident=alerts,
        ))
        for local_graph_path in sorted((scenario_dir / "local_graphs").glob("*.json")):
            local_graph = _load_json(local_graph_path)
            packet_local = {
                "scenario_id": packet["scenario_id"],
                "diagram_id": local_graph.get("diagram_id", local_graph_path.stem),
                "run_id": f"{scenario_dir.name}_{local_graph_path.stem}",
            }
            docs.extend(build_docs(packet_local, local_graph=local_graph, enterprise_graph=enterprise_graph))
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build InfraGraph local vector memory with ChromaDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--collection", default="infragraph_memory")
    parser.add_argument("--persist-dir", default="./runtime_state/vector_memory/chroma")
    parser.add_argument("--skip-v3-scenarios", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    _add_src(repo_root)
    try:
        from vector_memory.chroma_store import get_or_create_collection, upsert_documents
        from vector_memory.index_builder import build_vector_docs_from_graph_memory
    except RuntimeError as exc:
        print(str(exc))
        return 1

    docs: list[dict] = []
    docs.extend(_docs_from_live_ingestion(repo_root, build_vector_docs_from_graph_memory))
    docs.extend(_docs_from_incident_runs(repo_root, build_vector_docs_from_graph_memory))
    docs.extend(_docs_from_enterprise_gnn(repo_root, build_vector_docs_from_graph_memory))
    if not args.skip_v3_scenarios:
        docs.extend(_docs_from_v3_scenarios(repo_root, build_vector_docs_from_graph_memory))

    try:
        collection = get_or_create_collection(name=args.collection, persist_dir=args.persist_dir)
        indexed = upsert_documents(collection, docs)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    print("InfraGraph Vector Memory Build")
    print(f"  documents indexed: {indexed}")
    print(f"  collection:        {args.collection}")
    print(f"  persist path:      {args.persist_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
