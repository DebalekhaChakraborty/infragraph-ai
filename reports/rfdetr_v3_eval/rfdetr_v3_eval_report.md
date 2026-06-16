# RF-DETR V3 Detector Evaluation
**Status:** completed

RF-DETR detector evaluated against V3 verified annotations on 30 diagrams. Metrics are prototype benchmark metrics, not production accuracy.

---

> **Honest note:** Detector accuracy should only be claimed from strict class-aware metrics after class mapping is validated. Localization metrics indicate box quality independent of class labels.

---

## A. Strict Class-Aware Metrics
| Metric | Value |
|--------|-------|
| Global TP | 0 |
| Global FP | 265 |
| Global FN | 265 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | N/A |
| Mean AP@0.5 | 0.0000 |
| Mean IoU (matched) | N/A |
| Avg inference runtime | 19752.8000 ms |

### Per-Class Strict Metrics
| Class | TP | FP | FN | GT | Precision | Recall | F1 | AP@0.5 |
|-------|----|----|----|----|-----------|--------|----|--------|
| cloud_or_wan | 0 | 39 | 23 | 23 | 0.0000 | 0.0000 | N/A | 0.0000 |
| cls0 | 0 | 36 | 0 | 0 | 0.0000 | N/A | N/A | N/A |
| database | 0 | 6 | 12 | 12 | 0.0000 | 0.0000 | N/A | 0.0000 |
| firewall | 0 | 84 | 26 | 26 | 0.0000 | 0.0000 | N/A | 0.0000 |
| load_balancer | 0 | 23 | 6 | 6 | 0.0000 | 0.0000 | N/A | 0.0000 |
| router | 0 | 39 | 36 | 36 | 0.0000 | 0.0000 | N/A | 0.0000 |
| server | 0 | 12 | 84 | 84 | 0.0000 | 0.0000 | N/A | 0.0000 |
| service | 0 | 0 | 39 | 39 | N/A | 0.0000 | N/A | 0.0000 |
| switch | 0 | 26 | 39 | 39 | 0.0000 | 0.0000 | N/A | 0.0000 |

## B. Class-Agnostic Localization Metrics (IoU Only)
| Metric | Value |
|--------|-------|
| Localization TP | 265 |
| Localization FP | 0 |
| Localization FN | 0 |
| Localization Precision | 1.0000 |
| Localization Recall | 1.0000 |
| Localization F1 | 1.0000 |
| Localization Mean IoU | 0.9756 |

## C. Class-ID Shift Diagnostic
| Shift | TP | FP | FN | Precision | Recall | F1 | AP@0.5 |
|-------|----|----|----|-----------|---------|----|--------|
| 0 (original) | 0 | 265 | 265 | 0.0000 | 0.0000 | N/A | 0.0000 |
| +1 **(best)** | 265 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| -1 | 0 | 265 | 265 | 0.0000 | 0.0000 | N/A | 0.0000 |

**Diagnostic note:** RF-DETR runtime appears to return 0-indexed class IDs while COCO annotations are 1-indexed. +1 class-ID shift improves strict metrics significantly. Verify by inspecting per-class detection outputs vs. GT labels before claiming accuracy.

## D. Confusion Matrix Summary (IoU >= 0.5, class-agnostic matches)
| Predicted | GT Label | Count |
|-----------|----------|-------|
| cloud_or_wan | service | 39 |
| cls0 | router | 36 |
| database | load_balancer | 6 |
| firewall | server | 84 |
| load_balancer | cloud_or_wan | 23 |
| router | switch | 39 |
| server | database | 12 |
| switch | firewall | 26 |

### First 20 Matched Pairs
| Image | Predicted | GT | Confidence | IoU |
|-------|-----------|-----|------------|-----|
| enterprise_v3_0064__branch_topology.png | switch | firewall | 0.9631 | 0.9695 |
| enterprise_v3_0064__branch_topology.png | cls0 | router | 0.9541 | 0.9682 |
| enterprise_v3_0064__branch_topology.png | load_balancer | cloud_or_wan | 0.9518 | 0.9631 |
| enterprise_v3_0064__branch_topology.png | firewall | server | 0.9504 | 0.9667 |
| enterprise_v3_0064__branch_topology.png | firewall | server | 0.9496 | 0.9683 |
| enterprise_v3_0064__branch_topology.png | router | switch | 0.9494 | 0.981 |
| enterprise_v3_0064__branch_topology.png | firewall | server | 0.9484 | 0.9869 |
| enterprise_v3_0064__branch_topology.png | firewall | server | 0.9473 | 0.9628 |
| enterprise_v3_0064__branch_topology.png | firewall | server | 0.9468 | 0.9824 |
| enterprise_v3_0064__datacenter_topology.png | cls0 | router | 0.9647 | 0.9766 |
| enterprise_v3_0064__datacenter_topology.png | switch | firewall | 0.9602 | 0.9708 |
| enterprise_v3_0064__datacenter_topology.png | switch | firewall | 0.9571 | 0.9769 |
| enterprise_v3_0064__datacenter_topology.png | router | switch | 0.9548 | 0.9754 |
| enterprise_v3_0064__datacenter_topology.png | firewall | server | 0.9518 | 0.9643 |
| enterprise_v3_0064__datacenter_topology.png | firewall | server | 0.9516 | 0.9741 |
| enterprise_v3_0064__datacenter_topology.png | firewall | server | 0.9511 | 0.9728 |
| enterprise_v3_0064__datacenter_topology.png | router | switch | 0.9502 | 0.9768 |
| enterprise_v3_0064__datacenter_topology.png | router | switch | 0.9494 | 0.9766 |
| enterprise_v3_0064__datacenter_topology.png | firewall | server | 0.9492 | 0.96 |
| enterprise_v3_0064__datacenter_topology.png | firewall | server | 0.942 | 0.9602 |
