# Fixed signature-scrub v3 analysis

Integrity checks: **PASS**. Paired tasks: 154; valid sensitivity denominator: 150.

## Metrics (all tasks)

| Arm | pass@1 | pass@5 | pass@10 | aligned JIT compile@1 | aligned JIT compile@5 | compiled candidates |
|---|---:|---:|---:|---:|---:|---:|
| comparator | 17.6623% | 24.1136% | 27.2727% | 77.0779% | 93.8028% | 1187/1540 |
| neutral_exact | 0.7143% | 2.7623% | 4.5455% | 84.6104% | 97.3871% | 1303/1540 |
| name_only | 0.0000% | 0.0000% | 0.0000% | 3.1818% | 11.7708% | 49/1540 |

## Paired outcomes versus comparator

### neutral_exact

| Metric | Mean delta | metric gains/losses/ties | solved-task gains/losses/ties |
|---|---:|---:|---:|
| pass_at_1 | -16.9481 pp | 0/42/112 | 0/35/119 |
| pass_at_5 | -21.3513 pp | 0/42/112 | 0/35/119 |
| pass_at_10 | -22.7273 pp | 0/35/119 | 0/35/119 |
| compile_at_1 | +7.5325 pp | 59/61/34 | 3/0/151 |
| compile_at_5 | +3.5843 pp | 28/1/125 | 3/0/151 |

Static `fn0` shape: 1537/1540 candidates define it at top level; 1537 match the hidden arity.

### name_only

| Metric | Mean delta | metric gains/losses/ties | solved-task gains/losses/ties |
|---|---:|---:|---:|
| pass_at_1 | -17.6623 pp | 0/42/112 | 0/42/112 |
| pass_at_5 | -24.1136 pp | 0/42/112 | 0/42/112 |
| pass_at_10 | -27.2727 pp | 0/42/112 | 0/42/112 |
| compile_at_1 | -73.8961 pp | 2/146/6 | 1/120/33 |
| compile_at_5 | -82.0321 pp | 2/146/6 | 1/120/33 |

Static `fn0` shape: 1135/1540 candidates define it at top level; 272 match the hidden arity.

## Sensitivity metrics (valid tasks only; n=150)

This sensitivity excludes 4 inherited benchmark contract defects; the all-task results above remain primary.

| Arm | pass@1 | pass@5 | pass@10 | aligned JIT compile@1 | aligned JIT compile@5 | compiled candidates |
|---|---:|---:|---:|---:|---:|---:|
| comparator | 18.1333% | 24.7566% | 28.0000% | 79.1333% | 96.3042% | 1187/1500 |
| neutral_exact | 0.7333% | 2.8360% | 4.6667% | 86.8667% | 99.9841% | 1303/1500 |
| name_only | 0.0000% | 0.0000% | 0.0000% | 3.2667% | 12.0847% | 49/1500 |

## Sensitivity paired outcomes versus comparator (valid tasks only)

### neutral_exact (valid tasks)

| Metric | Mean delta | metric gains/losses/ties | solved-task gains/losses/ties |
|---|---:|---:|---:|
| pass_at_1 | -17.4000 pp | 0/42/108 | 0/35/115 |
| pass_at_5 | -21.9206 pp | 0/42/108 | 0/35/115 |
| pass_at_10 | -23.3333 pp | 0/35/115 | 0/35/115 |
| compile_at_1 | +7.7333 pp | 59/61/30 | 3/0/147 |
| compile_at_5 | +3.6799 pp | 28/1/121 | 3/0/147 |

### name_only (valid tasks)

| Metric | Mean delta | metric gains/losses/ties | solved-task gains/losses/ties |
|---|---:|---:|---:|
| pass_at_1 | -18.1333 pp | 0/42/108 | 0/42/108 |
| pass_at_5 | -24.7566 pp | 0/42/108 | 0/42/108 |
| pass_at_10 | -28.0000 pp | 0/42/108 | 0/42/108 |
| compile_at_1 | -75.8667 pp | 2/146/2 | 1/120/29 |
| compile_at_5 | -84.2196 pp | 2/146/2 | 1/120/29 |

## Integrity/provenance

Pairing: `normalized_hidden_test_sha256`. Checkpoint, generation seed, and every available scorer/SDK identifier agree across arms. Hidden tests agree for every pair after target-name normalization.
