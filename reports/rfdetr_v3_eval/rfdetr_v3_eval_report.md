# RF-DETR V3 Detector Evaluation

**Status:** completed


RF-DETR detector evaluated against V3 verified annotations on 30 diagrams. Metrics are prototype benchmark metrics, not production accuracy.


## Overall Metrics

| Metric | Value |
|--------|-------|
| precision | 0.0000 |
| recall | 0.0000 |
| f1 | N/A |
| mean_iou_matched | N/A |
| mean_ap_at_50 | 0.0000 |
| avg_inference_runtime_ms | 19712.3667 |
| global_tp | 0 |
| global_fp | 265 |
| global_fn | 265 |

## Per-Class Metrics

| Class | TP | FP | FN | Precision | Recall | F1 | AP@0.5 |
|-------|----|----|-----|-----------|--------|----|--------|
| cloud_or_wan | 0 | 39 | 23 | 0.0000 | 0.0000 | N/A | 0.0000 |
| cls0 | 0 | 36 | 0 | 0.0000 | N/A | N/A | N/A |
| database | 0 | 6 | 12 | 0.0000 | 0.0000 | N/A | 0.0000 |
| firewall | 0 | 84 | 26 | 0.0000 | 0.0000 | N/A | 0.0000 |
| load_balancer | 0 | 23 | 6 | 0.0000 | 0.0000 | N/A | 0.0000 |
| router | 0 | 39 | 36 | 0.0000 | 0.0000 | N/A | 0.0000 |
| server | 0 | 12 | 84 | 0.0000 | 0.0000 | N/A | 0.0000 |
| service | 0 | 0 | 39 | N/A | 0.0000 | N/A | 0.0000 |
| switch | 0 | 26 | 39 | 0.0000 | 0.0000 | N/A | 0.0000 |
