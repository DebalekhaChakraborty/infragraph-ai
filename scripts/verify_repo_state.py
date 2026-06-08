"""
Verify InfraGraph AI repository state before submission.
Uses only the Python standard library.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"
OK   = "\033[32m OK \033[0m"

failures = 0
warnings = 0


def check(label, result, *, warn=False):
    global failures, warnings
    if result:
        print(f"  [{OK}] {label}")
    elif warn:
        warnings += 1
        print(f"  [{WARN}] {label}")
    else:
        failures += 1
        print(f"  [{FAIL}] {label}")


def count_images(split_dir):
    if not os.path.isdir(split_dir):
        return 0
    return sum(
        1 for f in os.listdir(split_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / (1024 * 1024)


print("\n" + "=" * 60)
print("  InfraGraph AI — Repository State Verification")
print("=" * 60)

# ── Dataset ──────────────────────────────────────────────────
print("\n[Dataset]")
ds = os.path.join(ROOT, "datasets", "infragraph_v1")
check("datasets/infragraph_v1 exists", os.path.isdir(ds))

for split in ("train", "val", "test"):
    img_dir = os.path.join(ds, "images", split)
    n = count_images(img_dir)
    check(f"  images/{split}: {n} images", n > 0)

yaml_ok = os.path.isfile(os.path.join(ds, "dataset.yaml"))
check("  dataset.yaml present", yaml_ok)

# ── Model weights ─────────────────────────────────────────────
print("\n[Model weights]")
weights_dir = os.path.join(ROOT, "training_runs", "infragraph_yolo_v1", "weights")
best = os.path.join(weights_dir, "best.pt")
last = os.path.join(weights_dir, "last.pt")
check("training_runs/infragraph_yolo_v1/weights/best.pt", os.path.isfile(best))
check("training_runs/infragraph_yolo_v1/weights/last.pt", os.path.isfile(last))

if os.path.isfile(best):
    sz = os.path.getsize(best) / (1024 * 1024)
    check(f"  best.pt size > 1 MB ({sz:.1f} MB)", sz > 1)

# ── Stale / duplicate artifacts (warnings) ───────────────────
print("\n[Stale artifacts]")
stale = [
    ("runs/",                          os.path.join(ROOT, "runs")),
    ("data_generator/infragraph_dataset/", os.path.join(ROOT, "data_generator", "infragraph_dataset")),
    ("infragraph-ai.tar.gz",           os.path.join(ROOT, "infragraph-ai.tar.gz")),
]
for label, path in stale:
    present = os.path.exists(path)
    check(f"'{label}' absent (stale artifact)", not present, warn=True)

# ── Repo size ─────────────────────────────────────────────────
print("\n[Repository size]")
total_mb = dir_size_mb(ROOT)
print(f"  Approximate total size: {total_mb:.1f} MB")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
if failures == 0 and warnings == 0:
    print(f"  Overall: [{PASS}] All checks passed.")
elif failures == 0:
    print(f"  Overall: [{PASS}] Passed with {warnings} warning(s).")
else:
    print(f"  Overall: [{FAIL}] {failures} failure(s), {warnings} warning(s).")
print("=" * 60 + "\n")

sys.exit(failures)
