# Paper results

Values follow the final manuscript. Macro-F1 is reported in percent; gains are percentage points.

## Main comparison — Table 2

### Teacher targets

| Method | CUHK-X | MMAct | MM-Fit |
|---|---:|---:|---:|
| Equal-weight average | 72.1 | — | — |
| Direct | 75.0 | 87.8 | 94.6 |
| Naive Viterbi | 76.7 | 99.0 | 94.0 |
| Online filtering | 77.4 | 99.0 | 50.5 |
| PRISM | **80.0** | **99.0** | 94.0 |

Source: paper Table 2(a).

### Deployed students

| Method | CUHK-X | MMAct | MM-Fit |
|---|---:|---:|---:|
| From Direct | 75.0 | 87.8 | 94.6 |
| C0 (no adaptation) | 54.9 | 89.3 | 94.4 |
| Tent | 38.1 | — | — |
| Confidence PL | 64.6 | 83.4 | 94.7 |
| KD-soft | 74.7 | 87.4 | 94.7 |
| PRISM-D | **79.9** | **99.0** | 94.0 |

Source: paper Table 2(b). CUHK-X reports one macro-F1 pooled over all three outer folds.

On MM-Fit, where the clip-level estimate is already saturated, the gate keeps relational targets within 0.6 points of the baseline.

## CUHK-X gains by outer fold — Figure 3

| Outer fold | Clips | Selected teacher target gain over Direct |
|---|---:|---:|
| 0 | 973 | +4.0 |
| 1 | 663 | +3.6 |
| 2 | 834 | +7.0 |

Source: paper Figure 3 and Section V-A. These are teacher target gains.

## Computational cost — Table 4

Measurements use one laptop with an RTX 5070 Ti and 32 GB RAM. Offline times exclude expert training and C0 training.

| Step | Data | Time | Memory |
|---|---|---:|---:|
| Build targets | MMAct, 1,083 intervals | 0.35 s | 0.9 GB RAM |
| Direct student | CUHK-X, 834 clips | 48.3 s | 1.4 GB GPU |
| D→S student | CUHK-X, 834 clips | 46.3 s | 1.4 GB GPU |
| Full-chain student | CUHK-X, 834 clips | 46.4 s | 1.4 GB GPU |
| Selected student | MM-Fit, 11,668 windows | 587 s | — |
| PRISM-D deployment | Per clip | One forward pass | 23.6M parameters |

Source: paper Table 4. Deployment runs the student only.
