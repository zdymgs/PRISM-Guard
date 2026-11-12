# Paper results

Values follow the final manuscript. Macro-F1 is reported in percent; gains are percentage points. MMAct uses the official cross-subject split (evaluation on subjects 17–20) and MM-Fit the official test split.

## Main comparison — Table IV

### Teacher targets

| Method | CUHK-X | MMAct | MM-Fit |
|---|---:|---:|---:|
| Equal-weight average | 72.1 | 88.6 | 94.4 |
| Direct | 75.0 | 88.5 | 94.6 |
| Online filtering | 77.4 | 99.9 | 50.5 |
| Naive Viterbi | 76.7 | 99.9 | 94.0 |
| PRISM-Guard | **79.9** | **99.9** | 94.0 |

Source: paper Table IV(a). The equal-weight average is the plain mean of all expert probabilities in the bank (23 experts on CUHK-X, 15 on MMAct, 6 on MM-Fit). On MMAct and MM-Fit the decoder of PRISM-Guard is first order, so naive Viterbi coincides with D→S.

### Deployed students

| Method | CUHK-X | MMAct | MM-Fit |
|---|---:|---:|---:|
| From Direct | 75.0 | 88.5 | 94.6 |
| C0 (no adaptation) | 54.9 | 87.5 | 94.4 |
| KD-soft | 74.7 | 88.6 | 94.7 |
| Confidence PL | 64.6 | 83.3 | 94.7 |
| FixMatch | 51.7 | 89.5 | 92.0 |
| PRISM-Guard-D | **79.9** | **99.9** | 94.0 |

Source: paper Table IV(b). CUHK-X reports one macro-F1 pooled over all three outer folds. PRISM-Guard-D improves on the Direct student by 4.9 points on CUHK-X and 11.4 on MMAct.

On MM-Fit, where the clip-level estimate is already saturated, the gate keeps relational targets within 0.6 points of the baseline.

## CUHK-X gains by outer fold — Figure 3

| Outer fold | Clips | Selected teacher target gain over Direct |
|---|---:|---:|
| 0 | 973 | +4.0 |
| 1 | 663 | +3.6 |
| 2 | 834 | +7.0 |

Source: paper Figure 3 and Section VI-B. These are teacher target gains.

## Decoder sensitivity — Table V

| Dataset | Constant | Values (final) | Macro-F1 | Range | Lowest Δ vs Direct |
|---|---|---|---|---|---|
| CUHK-X (Direct 74.97) | λ | 0.25–0.65 (0.45) | 79.21–80.01 | 0.80 | +4.24 |
| CUHK-X | Back-off | 0.25–5 (1) | 79.24–79.95 | 0.71 | +4.27 |
| CUHK-X | Gap (s) | 15–60 (30) | 78.24–80.01 | 1.77 | +3.27 |
| MMAct (Direct 88.55) | λ | 0.15–0.75 (0.45) | 99.92 | 0.00 | +11.37 |
| MMAct | Smoothing | 0.25–4 (1) | 99.92 | 0.00 | +11.37 |
| MM-Fit (Direct 94.60) | λ | 0.05–0.45 (0.1) | 93.93–94.47 | 0.54 | −0.67 |
| MM-Fit | Smoothing | 0.25–4 (1) | 93.98–94.02 | 0.04 | −0.62 |
| MM-Fit | Gap (s) | 0.5–5 (1.5) | 94.02–94.47 | 0.45 | −0.58 |

Source: paper Table V; scripts in [experiments/sensitivity](experiments/sensitivity/README.md).

## Computational cost — Table VI

Measurements use one laptop with an RTX 5070 Ti and 32 GB RAM. Offline times exclude expert training and C0 training.

| Step | Data | Time | Memory |
|---|---|---:|---:|
| Build targets | MMAct, 1,116 intervals | 0.33 s | 0.9 GB RAM |
| Direct student | CUHK-X, 834 clips | 48.3 s | 1.5 GB GPU |
| D→S student | CUHK-X, 834 clips | 46.3 s | 1.5 GB GPU |
| Full-chain student | CUHK-X, 834 clips | 46.4 s | 1.5 GB GPU |
| Selected student | MM-Fit, 11,668 windows | 587 s | — |
| PRISM-Guard-D deployment | Per clip | One forward pass | 23.6M parameters |

Source: paper Table VI. Deployment runs the student only.
