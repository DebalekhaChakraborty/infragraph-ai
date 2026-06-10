"""
scripts/build_presentation_assets.py

Builds the product-facing asset layer (assets/gallery/ and assets/onboarding/)
from existing raw training/evaluation datasets without moving or deleting them.

What it does:
  1. Selects 15-20 curated V3 samples for the live onboarding flow.
     Prefers test/ > val/ > train/, diverse across all 5 diagram types.
     Copies source files into assets/onboarding/ONB-XXX/.
  2. Builds assets/gallery/manifest.json from V3 + V1/V2 datasets.
  3. Builds assets/onboarding/manifest.json.

Naming conventions:
  Gallery items:    DG-0001, DG-0002, ...
  Onboarding items: ONB-001, ONB-002, ...

Usage:
  python scripts/build_presentation_assets.py
  python scripts/build_presentation_assets.py --max-onboarding-samples 20 --max-gallery-items 250
  python scripts/build_presentation_assets.py --dry-run
  python scripts/build_presentation_assets.py --force   # overwrite existing onboarding dirs
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_infragraph_v3_root(repo_root: Path) -> Path:
    preferred = repo_root / "datasets" / "infragraph_v3"
    legacy = repo_root / "datasets" / "diagram_v3_enterprise"
    if preferred.exists():
        return preferred
    return legacy

# ---------------------------------------------------------------------------
# Display name tables
# ---------------------------------------------------------------------------

_V3_DISPLAY_NAMES: dict[str, str] = {
    "branch_topology":          "Branch Office Topology",
    "wan_topology":             "WAN Core Topology",
    "datacenter_topology":      "Data Center Topology",
    "app_db_topology":          "Application & Database Tier",
    "shared_services_topology": "Shared Services Topology",
}

_V3_DESCRIPTIONS: dict[str, str] = {
    "branch_topology":
        "Branch office network with edge routing and upstream WAN connectivity",
    "wan_topology":
        "Wide-area network backbone with MPLS circuits and internet peering paths",
    "datacenter_topology":
        "Data center core fabric with redundant switching and server clusters",
    "app_db_topology":
        "Application and database tier with load balancing and caching layers",
    "shared_services_topology":
        "Shared enterprise services including DNS, NTP, and management infrastructure",
}

_ONBOARD_PREFIX: dict[str, str] = {
    "branch_topology":          "New Branch Office Topology",
    "wan_topology":             "New WAN Core Topology",
    "datacenter_topology":      "New Data Center Topology",
    "app_db_topology":          "New Application & Database Tier",
    "shared_services_topology": "New Shared Services Topology",
}

_SPLITS_PREFERENCE = ("test", "val", "train")
_V3_DIAGRAM_TYPES = (
    "branch_topology", "wan_topology", "datacenter_topology",
    "app_db_topology", "shared_services_topology",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel(path: Path | str) -> str:
    """Return path relative to REPO_ROOT as forward-slash string, or '' if missing."""
    try:
        p = Path(path)
        return p.relative_to(REPO_ROOT).as_posix() if p.exists() else ""
    except ValueError:
        return str(path)


def _abs_or_empty(rel: str) -> str:
    """Resolve a relative-to-repo-root path string to absolute, or return '' if empty."""
    return str(REPO_ROOT / rel) if rel else ""


def _find_detected_preview_v3(scenario_id: str, diagram_id: str) -> str:
    """Return relative path to best available detection preview for a V3 diagram."""
    candidates = [
        REPO_ROOT / "outputs" / "live_ingestion"
        / f"{scenario_id}__{diagram_id}" / "detected.png",
        REPO_ROOT / "outputs" / "rfdetr_v3_predictions"
        / f"{scenario_id}__{diagram_id}.png",
    ]
    for p in candidates:
        if p.exists():
            return _rel(p)
    return ""


def _find_detected_preview_v1v2(dataset: str, split: str, diagram_id: str) -> str:
    """Return relative path to best available detection preview for V1/V2."""
    ds_tag = "v1" if "v1" in dataset else "v2"
    candidates = [
        REPO_ROOT / "outputs" / f"rfdetr_{ds_tag}" / split / diagram_id / "detected.png",
        REPO_ROOT / "outputs" / f"{ds_tag}_test_predictions_cpu" / f"{diagram_id}.png",
        REPO_ROOT / "outputs" / f"{ds_tag}_test_predictions_cpu" / f"{diagram_id}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return _rel(p)
    return ""


def _has_ocr_data(ann_path: Path) -> bool:
    """True if annotation JSON contains OCR text_blocks."""
    if not ann_path.exists():
        return False
    try:
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        return bool(ann.get("text_blocks"))
    except Exception:
        return False


def _has_connector_data(ann_path: Path) -> bool:
    """True if annotation JSON contains connectors."""
    if not ann_path.exists():
        return False
    try:
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        return bool(ann.get("connectors"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Onboarding sample selection and copy
# ---------------------------------------------------------------------------

def _select_onboarding_samples(v3_scenarios_root: Path, max_samples: int) -> list[dict]:
    """
    Select diverse V3 samples for the onboarding flow.

    Strategy: iterate splits in preference order (test → val → train).
    For each split, iterate scenarios.  For each scenario, try all 5 diagram
    types.  Add a sample for a type only if that type's quota is not yet full
    (quota = max_samples // 5).  This ensures even distribution across types.
    """
    per_type_quota = max_samples // len(_V3_DIAGRAM_TYPES)
    type_counts: dict[str, int] = {t: 0 for t in _V3_DIAGRAM_TYPES}
    samples: list[dict] = []

    for split in _SPLITS_PREFERENCE:
        split_dir = v3_scenarios_root / split
        if not split_dir.exists():
            continue

        for scen_dir in sorted(split_dir.iterdir()):
            if not scen_dir.is_dir():
                continue
            scen_id = scen_dir.name

            for dtype in _V3_DIAGRAM_TYPES:
                if type_counts[dtype] >= per_type_quota:
                    continue
                img_p = scen_dir / "diagrams" / f"{dtype}.png"
                if not img_p.exists():
                    continue
                ann_p = scen_dir / "annotations"  / f"{dtype}.json"
                lg_p  = scen_dir / "local_graphs"  / f"{dtype}.json"
                eg_p  = scen_dir / "enterprise_graph.json"
                sm_p  = scen_dir / "stitch_map.json"
                al_p  = scen_dir / "alerts.json"

                # require at minimum: image + annotation + local_graph
                if not (ann_p.exists() and lg_p.exists()):
                    continue

                samples.append({
                    "source_split":       split,
                    "source_scenario_id": scen_id,
                    "source_scenario_dir":str(scen_dir),
                    "source_diagram_id":  dtype,
                    "src_image":          img_p,
                    "src_annotation":     ann_p,
                    "src_local_graph":    lg_p,
                    "src_enterprise_graph": eg_p if eg_p.exists() else None,
                    "src_stitch_map":     sm_p if sm_p.exists() else None,
                    "src_alerts":         al_p if al_p.exists() else None,
                })
                type_counts[dtype] += 1

                if len(samples) >= max_samples:
                    return samples

    return samples


def _copy_onboarding_sample(
    sample: dict,
    onb_dir: Path,
    force: bool = False,
    log: logging.Logger | None = None,
) -> None:
    """Copy a V3 scenario's files into assets/onboarding/ONB-XXX/."""
    onb_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "original.png":          sample["src_image"],
        "annotation.json":       sample["src_annotation"],
        "local_graph.json":      sample["src_local_graph"],
        "enterprise_graph.json": sample.get("src_enterprise_graph"),
        "stitch_map.json":       sample.get("src_stitch_map"),
        "alerts.json":           sample.get("src_alerts"),
    }

    for dest_name, src_path in file_map.items():
        if src_path is None:
            continue
        dest = onb_dir / dest_name
        if dest.exists() and not force:
            continue
        try:
            shutil.copy2(src_path, dest)
        except Exception as exc:
            if log:
                log.warning(f"  Could not copy {src_path} → {dest}: {exc}")


# ---------------------------------------------------------------------------
# Onboarding manifest
# ---------------------------------------------------------------------------

def build_onboarding_manifest(
    samples: list[dict],
    onboarding_root: Path,
    log: logging.Logger | None = None,
) -> list[dict]:
    """Build and write assets/onboarding/manifest.json. Returns records list."""
    records: list[dict] = []

    for idx, sample in enumerate(samples, 1):
        sample_id = f"ONB-{idx:03d}"
        dtype     = sample["source_diagram_id"]
        onb_dir   = onboarding_root / sample_id
        rel_dir   = _rel(onb_dir) if onb_dir.exists() else f"assets/onboarding/{sample_id}"

        def _rp(fname: str) -> str:
            p = onb_dir / fname
            return f"{rel_dir}/{fname}" if p.exists() else ""

        records.append({
            "sample_id":          sample_id,
            "display_name":       _ONBOARD_PREFIX.get(dtype, f"New {dtype.replace('_',' ').title()}"),
            "diagram_type":       dtype,
            "status":             "not_onboarded",
            "source_dataset":     "v3",
            "source_split":       sample["source_split"],
            "source_scenario_id": sample["source_scenario_id"],
            "source_scenario_path": sample["source_scenario_dir"],
            "source_diagram_id":  dtype,
            "sample_dir":         rel_dir,
            "image_path":         _rp("original.png"),
            "annotation_path":    _rp("annotation.json"),
            "local_graph_path":   _rp("local_graph.json"),
            "enterprise_graph_path": _rp("enterprise_graph.json"),
            "stitch_map_path":    _rp("stitch_map.json"),
            "alerts_path":        _rp("alerts.json"),
        })

    manifest_path = onboarding_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    if log:
        log.info(f"  Onboarding manifest: {len(records)} samples -> {manifest_path}")
    return records


# ---------------------------------------------------------------------------
# Gallery manifest
# ---------------------------------------------------------------------------

def _v3_gallery_records(
    v3_scenarios_root: Path,
    max_items: int,
    counter_start: int = 1,
) -> list[dict]:
    """Yield gallery records for all V3 scenarios (test → val → train)."""
    records: list[dict] = []
    idx = counter_start

    for split in _SPLITS_PREFERENCE:
        split_dir = v3_scenarios_root / split
        if not split_dir.exists():
            continue
        for scen_dir in sorted(split_dir.iterdir()):
            if not scen_dir.is_dir():
                continue
            scen_id = scen_dir.name
            for dtype in _V3_DIAGRAM_TYPES:
                img_p  = scen_dir / "diagrams"     / f"{dtype}.png"
                if not img_p.exists():
                    continue

                ann_p  = scen_dir / "annotations"  / f"{dtype}.json"
                lg_p   = scen_dir / "local_graphs" / f"{dtype}.json"
                eg_p   = scen_dir / "enterprise_graph.json"
                sm_p   = scen_dir / "stitch_map.json"
                al_p   = scen_dir / "alerts.json"

                det_prev = _find_detected_preview_v3(scen_id, dtype)

                has_graph = lg_p.exists()
                has_ann   = ann_p.exists()
                has_conn  = _has_connector_data(ann_p) if has_ann else False
                has_ocr   = _has_ocr_data(ann_p)       if has_ann else False
                has_ent   = eg_p.exists()

                records.append({
                    "gallery_id":          f"DG-{idx:04d}",
                    "display_name":        _V3_DISPLAY_NAMES.get(dtype, dtype.replace("_", " ").title()),
                    "description":         _V3_DESCRIPTIONS.get(dtype, "Topology diagram"),
                    "status":              "available_in_graph_memory",
                    "source_dataset":      "v3",
                    "source_split":        split,
                    "source_scenario_id":  scen_id,
                    "source_diagram_id":   dtype,
                    "image_path":          _rel(img_p),
                    "annotation_path":     _rel(ann_p) if has_ann  else "",
                    "local_graph_path":    _rel(lg_p)  if has_graph else "",
                    "enterprise_graph_path": _rel(eg_p) if has_ent else "",
                    "stitch_map_path":     _rel(sm_p)  if sm_p.exists() else "",
                    "alerts_path":         _rel(al_p)  if al_p.exists() else "",
                    "preview_path":        "",
                    "detected_preview_path": det_prev,
                    "graph_metadata_available":    has_graph,
                    "connector_metadata_available":has_conn,
                    "ocr_metadata_available":      has_ocr,
                    "enterprise_mapping_available":has_ent,
                })
                idx += 1
                if len(records) >= max_items:
                    return records

    return records


def _v1v2_gallery_records(
    repo_root: Path,
    max_items: int,
    counter_start: int = 1,
) -> list[dict]:
    """Yield gallery records for V1 and V2 dataset diagrams."""
    records: list[dict] = []
    idx = counter_start

    for ds_tag, ds_name in [("v2", "infragraph_v2"), ("v1", "infragraph_v1")]:
        img_root = repo_root / "datasets" / ds_name / "images"
        if not img_root.exists():
            continue
        for split in _SPLITS_PREFERENCE:
            split_dir = img_root / split
            if not split_dir.exists():
                continue
            for img_p in sorted(split_dir.glob("*.png")):
                did = img_p.stem
                det_prev = _find_detected_preview_v1v2(ds_tag, split, did)

                # look for graph metadata from rfdetr pipeline
                pkt_p = (
                    repo_root / "outputs" / f"rfdetr_{ds_tag}"
                    / split / did / "graph_memory_packet.json"
                )
                eg_map_p = repo_root / "outputs" / f"rfdetr_{ds_tag}" / "enterprise_graph.json"

                has_graph = pkt_p.exists()
                has_ent   = eg_map_p.exists()

                records.append({
                    "gallery_id":   f"DG-{idx:04d}",
                    "display_name": f"Network Topology #{idx:04d}",
                    "description":  "Network topology diagram from training dataset",
                    "status":       "available_in_graph_memory",
                    "source_dataset":     ds_tag,
                    "source_split":       split,
                    "source_scenario_id": None,
                    "source_diagram_id":  did,
                    "image_path":              _rel(img_p),
                    "annotation_path":         "",
                    "local_graph_path":        _rel(pkt_p)   if has_graph else "",
                    "enterprise_graph_path":   _rel(eg_map_p) if has_ent  else "",
                    "stitch_map_path":         "",
                    "alerts_path":             "",
                    "preview_path":            "",
                    "detected_preview_path":   det_prev,
                    "graph_metadata_available":    has_graph,
                    "connector_metadata_available":False,
                    "ocr_metadata_available":      False,
                    "enterprise_mapping_available":has_ent,
                })
                idx += 1
                if len(records) >= max_items:
                    return records

    return records


def build_gallery_manifest(
    repo_root: Path,
    max_items: int,
    log: logging.Logger | None = None,
) -> list[dict]:
    """Build and write assets/gallery/manifest.json. Returns records list."""
    v3_scen_root = get_infragraph_v3_root(repo_root) / "scenarios"

    v3_records = _v3_gallery_records(v3_scen_root, max_items=max_items, counter_start=1)
    remain     = max(0, max_items - len(v3_records))

    # V1/V2 gallery items use DG IDs continuing after V3
    v1v2_records = _v1v2_gallery_records(
        repo_root, max_items=remain, counter_start=len(v3_records) + 1,
    ) if remain > 0 else []

    all_records = v3_records + v1v2_records

    gallery_dir = repo_root / "assets" / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = gallery_dir / "manifest.json"
    manifest_path.write_text(json.dumps(all_records, indent=2), encoding="utf-8")

    if log:
        log.info(
            f"  Gallery manifest: {len(all_records)} items "
            f"({len(v3_records)} V3 + {len(v1v2_records)} V1/V2) -> {manifest_path}"
        )
    return all_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build product-facing asset layer for InfraGraph AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-onboarding-samples", type=int, default=20,
        help="Maximum number of curated onboarding samples to prepare",
    )
    parser.add_argument(
        "--max-gallery-items", type=int, default=250,
        help="Maximum number of items in the gallery manifest",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing onboarding sample files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be built without writing any files",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log = _setup_logging(args.verbose)
    log.info("InfraGraph AI — Asset Build")
    log.info(f"  Repo root:               {REPO_ROOT}")
    log.info(f"  Max onboarding samples:  {args.max_onboarding_samples}")
    log.info(f"  Max gallery items:       {args.max_gallery_items}")
    log.info(f"  Dry run:                 {args.dry_run}")

    v3_scen_root = get_infragraph_v3_root(REPO_ROOT) / "scenarios"
    if not v3_scen_root.exists():
        log.error(
            f"V3 scenarios root not found: {v3_scen_root}\n"
            "Generate the dataset first:\n"
            "  python scripts/generate_infragraph_v3_dataset.py"
        )
        return 1

    # ── 1. Select onboarding samples ─────────────────────────────────────────
    log.info("\n[1/3] Selecting onboarding samples…")
    samples = _select_onboarding_samples(v3_scen_root, args.max_onboarding_samples)
    if not samples:
        log.error("No V3 samples found with required files (image + annotation + local_graph).")
        return 1

    log.info(f"  Selected {len(samples)} sample(s).")
    for i, s in enumerate(samples, 1):
        log.info(
            f"  ONB-{i:03d}  {s['source_split']:5s}  "
            f"{s['source_scenario_id']}  {s['source_diagram_id']}"
        )

    if not args.dry_run:
        onboarding_root = REPO_ROOT / "assets" / "onboarding"
        for idx, sample in enumerate(samples, 1):
            onb_dir = onboarding_root / f"ONB-{idx:03d}"
            _copy_onboarding_sample(sample, onb_dir, force=args.force, log=log)

        onb_records = build_onboarding_manifest(samples, onboarding_root, log=log)
        log.info(f"  {len(onb_records)} records -> assets/onboarding/manifest.json")
    else:
        log.info("  DRY RUN -- skipping file copy and manifest write.")

    # ── 2. Build gallery manifest ─────────────────────────────────────────────
    log.info("\n[2/3] Building gallery manifest…")
    if not args.dry_run:
        gallery_records = build_gallery_manifest(REPO_ROOT, args.max_gallery_items, log=log)
        log.info(f"  {len(gallery_records)} records -> assets/gallery/manifest.json")
    else:
        v3_count = sum(
            1
            for split in _SPLITS_PREFERENCE
            for scen_dir in sorted((v3_scen_root / split).iterdir())
            if scen_dir.is_dir()
            for dtype in _V3_DIAGRAM_TYPES
            if (scen_dir / "diagrams" / f"{dtype}.png").exists()
        ) if v3_scen_root.exists() else 0
        log.info(f"  DRY RUN — would include up to {min(v3_count, args.max_gallery_items)} V3 items")

    # ── 3. Summary ────────────────────────────────────────────────────────────
    log.info("\n[3/3] Done.")
    if not args.dry_run:
        log.info("  assets/gallery/manifest.json    - powers Diagram Gallery UI")
        log.info("  assets/onboarding/manifest.json - powers Onboard New Diagram UI")
        log.info("  assets/onboarding/ONB-*/        - curated sample file trees")
        log.info("")
        log.info("Next: streamlit run app/streamlit_app.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
