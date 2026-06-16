# RF-DETR Runtime Evidence
Generated: 2026-06-16T01:01:22.069685+00:00

---

> **This is RF-DETR runtime/inference evidence. Detector accuracy/mAP is not claimed here. The production demo uses RF-DETR-supported detection with verified annotation fallback for reliability.**

---

## Summary
| Field | Value |
|-------|-------|
| Images attempted | 5 |
| Successful inference runs | 5 |
| Average inference runtime | 19760 ms |
| Total detections (all images) | 45 |
| Checkpoint | `/workspace/shared/infragraph-ai/model_artifacts/rfdetr_v3/checkpoint_best_total.pth` |

## Per-Image Results
| Image | Status | Runtime (ms) | Detections | Annotated output |
|-------|--------|-------------|------------|------------------|
| `enterprise_v3_0064__datacenter_topology.png` | OK | 41386 | 11 | `enterprise_v3_0064__datacenter_topology_annotated.png` |
| `enterprise_v3_0065__app_db_topology.png` | OK | 44795 | 12 | `enterprise_v3_0065__app_db_topology_annotated.png` |
| `enterprise_v3_0066__shared_services_topology.png` | OK | 40379 | 8 | `enterprise_v3_0066__shared_services_topology_annotated.png` |
| `enterprise_v3_0067__branch_topology.png` | OK | 38826 | 7 | `enterprise_v3_0067__branch_topology_annotated.png` |
| `enterprise_v3_0068__wan_topology.png` | OK | 43679 | 7 | `enterprise_v3_0068__wan_topology_annotated.png` |

## Detection Summaries

### enterprise_v3_0064__datacenter_topology.png
- Inference runtime: 19891 ms
- Model class: RFDETRBase
- Checkpoint strategy: constructor_pretrain_weights
- Detections (11):
  - `cls0` confidence=0.965
  - `switch` confidence=0.960
  - `switch` confidence=0.957
  - `router` confidence=0.955
  - `firewall` confidence=0.952
  - `firewall` confidence=0.952
  - `firewall` confidence=0.951
  - `router` confidence=0.950
  - `router` confidence=0.949
  - `firewall` confidence=0.949

### enterprise_v3_0065__app_db_topology.png
- Inference runtime: 19617 ms
- Model class: RFDETRBase
- Checkpoint strategy: constructor_pretrain_weights
- Detections (12):
  - `cloud_or_wan` confidence=0.967
  - `server` confidence=0.962
  - `cloud_or_wan` confidence=0.959
  - `database` confidence=0.957
  - `cloud_or_wan` confidence=0.957
  - `server` confidence=0.956
  - `firewall` confidence=0.954
  - `firewall` confidence=0.950
  - `firewall` confidence=0.950
  - `router` confidence=0.949

### enterprise_v3_0066__shared_services_topology.png
- Inference runtime: 19865 ms
- Model class: RFDETRBase
- Checkpoint strategy: constructor_pretrain_weights
- Detections (8):
  - `cloud_or_wan` confidence=0.960
  - `firewall` confidence=0.958
  - `cloud_or_wan` confidence=0.956
  - `cloud_or_wan` confidence=0.953
  - `firewall` confidence=0.952
  - `firewall` confidence=0.951
  - `cloud_or_wan` confidence=0.950
  - `cloud_or_wan` confidence=0.950

### enterprise_v3_0067__branch_topology.png
- Inference runtime: 19539 ms
- Model class: RFDETRBase
- Checkpoint strategy: constructor_pretrain_weights
- Detections (7):
  - `switch` confidence=0.963
  - `firewall` confidence=0.952
  - `firewall` confidence=0.952
  - `firewall` confidence=0.951
  - `router` confidence=0.950
  - `firewall` confidence=0.949
  - `cls0` confidence=0.948

### enterprise_v3_0068__wan_topology.png
- Inference runtime: 19886 ms
- Model class: RFDETRBase
- Checkpoint strategy: constructor_pretrain_weights
- Detections (7):
  - `cls0` confidence=0.963
  - `cls0` confidence=0.963
  - `cls0` confidence=0.960
  - `load_balancer` confidence=0.956
  - `switch` confidence=0.955
  - `load_balancer` confidence=0.954
  - `load_balancer` confidence=0.953
