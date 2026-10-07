# Graph-v2 Seed-42 Architecture Interaction Study

This report replays archived candidate pools only. No model inference or Dart evaluation was run.

- Strict artifact/provenance validation: **PASS** for 15 cells.
- All effects are paired against seed-42 GCB no-GINE and show 95% task-bootstrap intervals.
- These N=154 cells are descriptive table-completion runs, below the decision floor.

## Encoder Family x Trainability

| Cell | pass@1 | pass@5 | pass@10 | compile@1 | compile@5 | checkpoint MiB | train runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| GCB no-GINE (trainable) | 0.1617 | 0.2314 | 0.2597 | 0.7831 | 0.9527 | 861.7 | n/a |
| GCB no-GINE (frozen) | 0.0974 | 0.1934 | 0.2208 | 0.4831 | 0.8468 | 855.0 | n/a |
| CLAP no-GINE (trainable) | 0.1656 | 0.2318 | 0.2532 | 0.7708 | 0.9501 | 882.1 | n/a |
| CLAP no-GINE (frozen) | 0.1532 | 0.2174 | 0.2338 | 0.7390 | 0.9334 | 855.0 | 85.5 min |

## Encoder Family x Graph Propagation

| Cell | pass@1 | pass@5 | pass@10 | compile@1 | compile@5 | checkpoint MiB | train runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| GCB no-GINE (trainable) | 0.1617 | 0.2314 | 0.2597 | 0.7831 | 0.9527 | 861.7 | n/a |
| GCB CFG-GINE (trainable) | 0.1734 | 0.2343 | 0.2597 | 0.7675 | 0.9460 | 861.7 | n/a |
| CLAP no-GINE (trainable) | 0.1656 | 0.2318 | 0.2532 | 0.7708 | 0.9501 | 882.1 | n/a |
| CLAP CFG-GINE (trainable) | 0.1273 | 0.2193 | 0.2468 | 0.6338 | 0.9020 | 882.1 | 102.1 min |

## Factor Contrasts

The interaction row is the paired difference-in-differences. Positive trainability interaction means freezing hurts CLAP less than GCB; negative propagation interaction means CFG-GINE hurts CLAP more than GCB.

| Study | Contrast | delta pass@1 | delta pass@5 | delta pass@10 | delta compile@1 |
|---|---|---:|---:|---:|---:|
| Family x trainability | Frozen - trainable, GCB | -0.0643 [-0.1006, -0.0299] | -0.0380 [-0.0697, -0.0083] | -0.0390 [-0.0779, -0.0065] | -0.3000 [-0.3500, -0.2519] |
| Family x trainability | Frozen - trainable, CLAP | -0.0123 [-0.0312, +0.0058] | -0.0144 [-0.0315, +0.0005] | -0.0195 [-0.0455, +0.0000] | -0.0318 [-0.0656, +0.0013] |
| Family x trainability | CLAP - GCB, trainable | +0.0039 [-0.0097, +0.0182] | +0.0004 [-0.0196, +0.0191] | -0.0065 [-0.0325, +0.0195] | -0.0123 [-0.0377, +0.0136] |
| Family x trainability | CLAP - GCB, frozen | +0.0558 [+0.0273, +0.0877] | +0.0240 [+0.0047, +0.0455] | +0.0130 [-0.0130, +0.0390] | +0.2558 [+0.2084, +0.3032] |
| Family x trainability | Interaction | +0.0519 [+0.0195, +0.0870] | +0.0236 [-0.0016, +0.0493] | +0.0195 [-0.0130, +0.0519] | +0.2682 [+0.2123, +0.3253] |
| Family x propagation | CFG-GINE - no-GINE, GCB | +0.0117 [-0.0045, +0.0299] | +0.0029 [-0.0234, +0.0271] | +0.0000 [-0.0325, +0.0325] | -0.0156 [-0.0442, +0.0123] |
| Family x propagation | CFG-GINE - no-GINE, CLAP | -0.0383 [-0.0656, -0.0136] | -0.0125 [-0.0375, +0.0113] | -0.0065 [-0.0390, +0.0260] | -0.1370 [-0.1825, -0.0935] |
| Family x propagation | CLAP - GCB, no-GINE | +0.0039 [-0.0097, +0.0188] | +0.0004 [-0.0188, +0.0193] | -0.0065 [-0.0325, +0.0195] | -0.0123 [-0.0383, +0.0136] |
| Family x propagation | CLAP - GCB, CFG-GINE | -0.0461 [-0.0792, -0.0169] | -0.0150 [-0.0433, +0.0137] | -0.0130 [-0.0519, +0.0260] | -0.1338 [-0.1825, -0.0870] |
| Family x propagation | Interaction | -0.0500 [-0.0831, -0.0188] | -0.0154 [-0.0514, +0.0189] | -0.0065 [-0.0519, +0.0390] | -0.1214 [-0.1773, -0.0662] |

## Paired Effects vs Baseline

| Variant | delta pass@1 | delta pass@5 | delta pass@10 | delta compile@1 | gains/losses at k=10 |
|---|---:|---:|---:|---:|---:|
| GCB no-GINE (trainable) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0/0 |
| Regions 4 | +0.0091 [-0.0052, +0.0247] | +0.0019 [-0.0167, +0.0212] | +0.0000 [-0.0260, +0.0260] | -0.0110 [-0.0390, +0.0162] | 2/2 |
| Regions 8 | -0.0013 [-0.0169, +0.0149] | -0.0087 [-0.0293, +0.0097] | -0.0195 [-0.0519, +0.0065] | -0.0253 [-0.0494, -0.0013] | 1/4 |
| Regions 16 | +0.0149 [-0.0026, +0.0357] | +0.0098 [-0.0128, +0.0329] | +0.0130 [-0.0195, +0.0455] | -0.0123 [-0.0409, +0.0156] | 4/2 |
| CLAP no-GINE (trainable) | +0.0039 [-0.0097, +0.0182] | +0.0004 [-0.0190, +0.0192] | -0.0065 [-0.0390, +0.0195] | -0.0123 [-0.0383, +0.0136] | 2/3 |
| CLAP no-GINE (frozen) | -0.0084 [-0.0312, +0.0162] | -0.0140 [-0.0402, +0.0098] | -0.0260 [-0.0584, +0.0000] | -0.0442 [-0.0753, -0.0123] | 1/5 |
| CLAP CFG-GINE (trainable) | -0.0344 [-0.0656, -0.0065] | -0.0121 [-0.0379, +0.0119] | -0.0130 [-0.0519, +0.0195] | -0.1494 [-0.1929, -0.1078] | 3/5 |
| Multivector 2 | -0.0169 [-0.0370, +0.0032] | -0.0034 [-0.0244, +0.0167] | -0.0065 [-0.0325, +0.0195] | -0.0877 [-0.1227, -0.0532] | 2/3 |
| Multivector 4 | -0.0032 [-0.0182, +0.0117] | +0.0049 [-0.0186, +0.0277] | +0.0000 [-0.0325, +0.0390] | -0.0182 [-0.0481, +0.0123] | 4/4 |
| Multivector 8 | -0.0039 [-0.0266, +0.0175] | +0.0039 [-0.0203, +0.0294] | -0.0130 [-0.0455, +0.0195] | -0.0708 [-0.1058, -0.0370] | 2/4 |
| No global attention | +0.0058 [-0.0091, +0.0214] | -0.0001 [-0.0194, +0.0175] | -0.0130 [-0.0390, +0.0130] | -0.0097 [-0.0351, +0.0156] | 1/3 |
| GINE 2 layers | -0.0013 [-0.0162, +0.0136] | +0.0057 [-0.0129, +0.0236] | +0.0000 [-0.0260, +0.0260] | -0.0123 [-0.0383, +0.0130] | 2/2 |
| No block position | +0.0130 [-0.0039, +0.0318] | -0.0028 [-0.0285, +0.0207] | -0.0195 [-0.0519, +0.0065] | -0.0026 [-0.0331, +0.0286] | 1/4 |
| GCB no-GINE (frozen) | -0.0643 [-0.1006, -0.0299] | -0.0380 [-0.0711, -0.0083] | -0.0390 [-0.0779, -0.0065] | -0.3000 [-0.3500, -0.2506] | 1/7 |

## Full Architecture Screen

| Variant | pass@1 | pass@5 | pass@10 | compile@1 | compile@5 | CodeBLEU | solved |
|---|---:|---:|---:|---:|---:|---:|---:|
| GCB no-GINE (trainable) | 0.1617 | 0.2314 | 0.2597 | 0.7831 | 0.9527 | 0.6726 | 40 |
| Regions 4 | 0.1708 | 0.2333 | 0.2597 | 0.7721 | 0.9439 | 0.6803 | 40 |
| Regions 8 | 0.1604 | 0.2227 | 0.2403 | 0.7578 | 0.9420 | 0.6732 | 37 |
| Regions 16 | 0.1766 | 0.2411 | 0.2727 | 0.7708 | 0.9380 | 0.6785 | 42 |
| CLAP no-GINE (trainable) | 0.1656 | 0.2318 | 0.2532 | 0.7708 | 0.9501 | 0.6715 | 39 |
| CLAP no-GINE (frozen) | 0.1532 | 0.2174 | 0.2338 | 0.7390 | 0.9334 | 0.6571 | 36 |
| CLAP CFG-GINE (trainable) | 0.1273 | 0.2193 | 0.2468 | 0.6338 | 0.9020 | 0.6612 | 38 |
| Multivector 2 | 0.1448 | 0.2279 | 0.2532 | 0.6955 | 0.9498 | 0.6679 | 39 |
| Multivector 4 | 0.1584 | 0.2362 | 0.2597 | 0.7649 | 0.9479 | 0.6670 | 40 |
| Multivector 8 | 0.1578 | 0.2353 | 0.2468 | 0.7123 | 0.9388 | 0.6689 | 38 |
| No global attention | 0.1675 | 0.2312 | 0.2468 | 0.7734 | 0.9422 | 0.6771 | 38 |
| GINE 2 layers | 0.1604 | 0.2370 | 0.2597 | 0.7708 | 0.9453 | 0.6695 | 40 |
| No block position | 0.1747 | 0.2286 | 0.2403 | 0.7805 | 0.9528 | 0.6752 | 37 |
| GCB no-GINE (frozen) | 0.0974 | 0.1934 | 0.2208 | 0.4831 | 0.8468 | 0.6557 | 34 |

## Block-Count Strata (pass@10)

| Variant | low | mid | high |
|---|---:|---:|---:|
| GCB no-GINE (trainable) | 0.3137 | 0.2941 | 0.1731 |
| Regions 4 | 0.3529 | 0.2549 | 0.1731 |
| Regions 8 | 0.2941 | 0.2745 | 0.1538 |
| Regions 16 | 0.3333 | 0.2941 | 0.1923 |
| CLAP no-GINE (trainable) | 0.3137 | 0.2745 | 0.1731 |
| CLAP no-GINE (frozen) | 0.2941 | 0.2353 | 0.1731 |
| CLAP CFG-GINE (trainable) | 0.3137 | 0.2549 | 0.1731 |
| Multivector 2 | 0.3137 | 0.2745 | 0.1731 |
| Multivector 4 | 0.3529 | 0.2353 | 0.1923 |
| Multivector 8 | 0.3137 | 0.2549 | 0.1731 |
| No global attention | 0.3137 | 0.2549 | 0.1731 |
| GINE 2 layers | 0.3137 | 0.2941 | 0.1731 |
| No block position | 0.3137 | 0.2745 | 0.1346 |
| GCB no-GINE (frozen) | 0.2941 | 0.2549 | 0.1154 |

## Representation and Artifact Cost

| Variant | representation | encoder | trainable | checkpoint MiB | candidate pool MiB |
|---|---|---|---:|---:|---:|
| GCB no-GINE (trainable) | CLS/block | GCB | yes | 861.7 | 0.88 |
| Regions 4 | CLS/block + regions/4 | GCB | yes | 882.1 | 0.96 |
| Regions 8 | CLS/block + regions/8 | GCB | yes | 882.1 | 1.07 |
| Regions 16 | CLS/block + regions/16 | GCB | yes | 882.1 | 1.02 |
| CLAP no-GINE (trainable) | CLS/block | CLAP | yes | 882.1 | 0.99 |
| CLAP no-GINE (frozen) | CLS/block | CLAP | no | 855.0 | 1.25 |
| CLAP CFG-GINE (trainable) | CLS/block + 4L-GINE/cfg | CLAP | yes | 882.1 | 0.82 |
| Multivector 2 | 2 query/block | GCB | yes | 882.1 | 1.29 |
| Multivector 4 | 4 query/block | GCB | yes | 882.1 | 1.03 |
| Multivector 8 | 8 query/block | GCB | yes | 882.1 | 1.58 |
| No global attention | CLS/block + no-attn | GCB | yes | 882.1 | 0.94 |
| GINE 2 layers | CLS/block + 2L-GINE/none | GCB | yes | 873.0 | 0.94 |
| No block position | CLS/block + no-pos | GCB | yes | 882.1 | 1.59 |
| GCB no-GINE (frozen) | CLS/block | GCB | no | 855.0 | 2.52 |

Runtime note: exact trainer runtime was available in the just-completed queue for frozen CLAP (85.5 min) and CLAP CFG-GINE (102.1 min). Older queue logs are not encoded in the seven result artifacts, so their runtime cells are intentionally reported as unavailable rather than inferred.
