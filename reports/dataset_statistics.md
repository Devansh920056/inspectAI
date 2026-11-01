# MVTec AD statistics report

Generated: 2026-11-01T09:08:20+00:00  
Data root: `data/mvtec_ad`

## Summary

| Category | Train (normal) | Test normal | Test defective | Defect types | Image size(s) | Mean defect area | Problems |
|---|---:|---:|---:|---:|---|---:|---:|
| bottle | 209 | 20 | 63 | 3 | 900x900 | 7.62% | 0 |
| hazelnut | 391 | 40 | 70 | 4 | 1024x1024 | 3.35% | 0 |
| carpet | 280 | 28 | 89 | 5 | 1024x1024 | 2.11% | 0 |

## bottle

- Train: 209 images (all normal)
- Test: 83 images — 20 normal, 63 defective (75.9% defective)
- Defect types: broken_large (20), broken_small (22), contamination (21)
- Image formats: PNG (292); modes: RGB (292); sizes: 900x900 (292)
- Masks: 63 (formats PNG (63), modes L (63)), empty masks: 0

| Defect type | Images | Min area | Median | Mean | Max |
|---|---:|---:|---:|---:|---:|
| broken_large | 20 | 4.09% | 11.87% | 11.70% | 27.42% |
| broken_small | 22 | 0.88% | 2.44% | 3.07% | 8.06% |
| contamination | 21 | 0.58% | 7.25% | 8.49% | 16.53% |

- Train pixel mean (RGB, [0,1]): [0.53541, 0.53168, 0.54796]
- Train pixel std (RGB, [0,1]): [0.36676, 0.36667, 0.34992]

- Validation: OK (292 images fully decoded)

## hazelnut

- Train: 391 images (all normal)
- Test: 110 images — 40 normal, 70 defective (63.6% defective)
- Defect types: crack (18), cut (17), hole (18), print (17)
- Image formats: PNG (501); modes: RGB (501); sizes: 1024x1024 (501)
- Masks: 70 (formats PNG (70), modes L (70)), empty masks: 0

| Defect type | Images | Min area | Median | Mean | Max |
|---|---:|---:|---:|---:|---:|
| crack | 18 | 1.28% | 4.70% | 8.40% | 27.66% |
| cut | 17 | 0.40% | 0.79% | 0.88% | 2.02% |
| hole | 18 | 0.23% | 1.24% | 1.33% | 2.93% |
| print | 17 | 1.52% | 2.44% | 2.63% | 4.83% |

- Train pixel mean (RGB, [0,1]): [0.24006, 0.17684, 0.17134]
- Train pixel std (RGB, [0,1]): [0.16887, 0.07598, 0.04384]

- Validation: OK (501 images fully decoded)

## carpet

- Train: 280 images (all normal)
- Test: 117 images — 28 normal, 89 defective (76.1% defective)
- Defect types: color (19), cut (17), hole (17), metal_contamination (17), thread (19)
- Image formats: PNG (397); modes: RGB (397); sizes: 1024x1024 (397)
- Masks: 89 (formats PNG (89), modes L (89)), empty masks: 0

| Defect type | Images | Min area | Median | Mean | Max |
|---|---:|---:|---:|---:|---:|
| color | 19 | 0.35% | 1.57% | 1.87% | 6.27% |
| cut | 17 | 3.18% | 4.35% | 4.88% | 8.02% |
| hole | 17 | 0.67% | 1.84% | 2.25% | 5.64% |
| metal_contamination | 17 | 0.16% | 0.39% | 0.49% | 1.35% |
| thread | 19 | 0.18% | 0.64% | 1.18% | 5.32% |

- Train pixel mean (RGB, [0,1]): [0.36295, 0.35002, 0.35608]
- Train pixel std (RGB, [0,1]): [0.14507, 0.13995, 0.12768]

- Validation: OK (397 images fully decoded)
