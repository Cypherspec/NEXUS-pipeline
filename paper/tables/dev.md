| Candidate (vs. v1) | Mean rank | ΔR² (pp) | Mean rel. log loss | Median rel. log loss | Wins / 16 | Rel. error change |
|---|---|---|---|---|---|---|
| +extended portfolio +L2 stack (3) | 4.88 | +0.64 | +7.6% | +0.83% | 10 | +6.5% |
| +L2 stack {lgbm, logreg, catboost} | 5.34 | +0.64 | +7.5% | +0.30% | 9 | +7.6% |
| +L2 stack {lgbm, logreg} (v2) | 6.03 | +0.62 | +7.0% | +0.36% | 11 | +7.2% |
| +L2 stack {lgbm, catboost} | 5.09 | +0.48 | +7.2% | +0.31% | 11 | +6.9% |
| +L2 stack {lgbm} | 6.09 | +0.38 | +6.3% | +0.35% | 12 | +7.1% |
| repeated cross-fitting R=3 | 7.12 | +0.05 | -1.5% | +0.00% | 8 | -1.1% |
| repeated cross-fitting R=2 | 6.44 | +0.04 | -0.9% | +0.10% | 11 | -0.9% |
| +13-config extended portfolio | 6.03 | +0.03 | +0.1% | +0.40% | 9 | -0.6% |
| v1 (greedy selection + temperature) | 7.34 | +0.00 | +0.0% | +0.00% | 0 | +0.0% |
| convex weight optimisation | 7.44 | -0.04 | -0.0% | +0.00% | 8 | -0.3% |
| early-stopped GBDTs (swap) | 7.75 | -0.22 | -5.3% | -0.25% | 7 | -1.7% |
| bagged ensemble selection | 8.44 | -0.49 | -10.5% | -0.49% | 5 | -0.9% |
