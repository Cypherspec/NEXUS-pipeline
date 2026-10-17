
### bace  (1513 molecules, 1 task(s), roc_auc, higher is better)
Reference D-MPNN (Yang et al. 2019, scaffold split): 0.857

| Method | mean | std | seeds |
|---|---|---|---|
| ECFP+descriptors + RF | 0.9056 | 0.0107 | 3 |
| ECFP+descriptors + linear | 0.9000 | 0.0187 | 3 |
| ECFP + linear | 0.8980 | 0.0171 | 3 |
| ECFP + RF | 0.8947 | 0.0052 | 3 |
| D-MPNN + descriptors | 0.8910 | 0.0103 | 3 |
| descriptors + RF | 0.8641 | 0.0261 | 3 |
| D-MPNN | 0.8526 | 0.0152 | 3 |
| descriptors + linear | 0.7753 | 0.0037 | 3 |

### bbbp  (2050 molecules, 1 task(s), roc_auc, higher is better)
Reference D-MPNN (Yang et al. 2019, scaffold split): 0.913

| Method | mean | std | seeds |
|---|---|---|---|
| ECFP + RF | 0.9339 | 0.0112 | 3 |
| ECFP+descriptors + RF | 0.9249 | 0.0109 | 3 |
| ECFP+descriptors + linear | 0.9167 | 0.0113 | 3 |
| ECFP + linear | 0.9166 | 0.0123 | 3 |
| D-MPNN | 0.9109 | 0.0056 | 3 |
| descriptors + RF | 0.8908 | 0.0139 | 3 |
| D-MPNN + descriptors | 0.8907 | 0.0246 | 3 |
| descriptors + linear | 0.8441 | 0.0044 | 3 |

### clintox  (1484 molecules, 2 task(s), roc_auc, higher is better)
Reference D-MPNN (Yang et al. 2019, scaffold split): 0.897

| Method | mean | std | seeds |
|---|---|---|---|
| D-MPNN + descriptors | 0.8969 | 0.0621 | 3 |
| D-MPNN | 0.8962 | 0.0412 | 3 |
| ECFP+descriptors + linear | 0.8236 | 0.0465 | 3 |
| ECFP + linear | 0.7969 | 0.0352 | 3 |
| descriptors + RF | 0.7760 | 0.0174 | 3 |
| descriptors + linear | 0.7709 | 0.1095 | 3 |
| ECFP+descriptors + RF | 0.7192 | 0.0390 | 3 |
| ECFP + RF | 0.6954 | 0.0548 | 3 |