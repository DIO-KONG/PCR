# Scale Gate Analysis

Run: `result/submap_walkforward/submap_327_align_only_se3`

## Candidate Gate

```text
base_icp_suspect = icp_fitness < 0.55 and conflict_ratio > 0.55 and accepted_ratio > 0.05
broad_accepted_region = accepted_xz_area > 25.0 and accepted_extent_max > 7.0
scale_shadow_gate_v1 = base_icp_suspect and broad_accepted_region
```

Rationale: the broad-region term separates submap_007 scale-shadow steps from early steps that also have low ICP fitness and high conflict, but whose accepted points are compact local additions.

## Hit Summary

- Fusion attempts analyzed: 184
- Gate hits: 16
- First 20 hits: 0 (none)
- Step 71-80 hits: 5 (72, 73, 75, 76, 77)
- Submap_007 hits: 5 (72, 73, 75, 76, 77)

## First 20 Impact

| step | submap | pose | fusion | p90 | fitness | accR | dupR | confR | xz_area | extent | gate |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | submap_000 | needs_review | needs_review | 0.420 | 0.433 | 0.162 | 0.356 | 0.482 | 8.3 | 2.9 |  |
| 2 | submap_000 | accepted | needs_review | 0.019 | 0.400 | 0.157 | 0.216 | 0.627 | 9.5 | 3.2 |  |
| 3 | submap_000 | accepted | needs_review | 0.016 | 0.448 | 0.127 | 0.169 | 0.704 | 9.4 | 3.6 |  |
| 4 | submap_000 | needs_review | needs_review | 0.152 | 0.412 | 0.139 | 0.091 | 0.770 | 2.2 | 1.5 |  |
| 5 | submap_000 | accepted | needs_review | 0.029 | 0.365 | 0.176 | 0.156 | 0.668 | 8.1 | 3.1 |  |
| 6 | submap_000 | accepted | needs_review | 0.026 | 0.416 | 0.001 | 0.278 | 0.721 | 3.2 | 1.9 |  |

## Step 71-80

| step | submap | pose | fusion | p90 | fitness | accR | dupR | confR | xz_area | extent | gate |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 71 | submap_007 | accepted | needs_review | 0.033 | 0.638 | 0.022 | 0.397 | 0.581 | 16.7 | 5.4 |  |
| 72 | submap_007 | accepted | needs_review | 0.066 | 0.539 | 0.112 | 0.178 | 0.710 | 45.6 | 8.4 | yes |
| 73 | submap_007 | accepted | needs_review | 0.094 | 0.442 | 0.105 | 0.128 | 0.767 | 81.5 | 10.1 | yes |
| 75 | submap_007 | needs_review | needs_review | 0.271 | 0.346 | 0.113 | 0.295 | 0.591 | 104.5 | 12.2 | yes |
| 76 | submap_007 | needs_review | needs_review | 0.347 | 0.395 | 0.089 | 0.338 | 0.573 | 109.4 | 12.4 | yes |
| 77 | submap_007 | needs_review | needs_review | 0.414 | 0.351 | 0.096 | 0.237 | 0.667 | 101.4 | 10.8 | yes |
| 78 | submap_007 | needs_review | needs_review | 0.171 | 0.539 | 0.010 | 0.293 | 0.697 | 76.8 | 9.4 |  |
| 80 | submap_007 | needs_review | needs_review | 0.419 | 0.620 | 0.038 | 0.367 | 0.595 | 30.3 | 7.2 |  |

## All Gate Hits

| step | submap | pose | fusion | p90 | fitness | accR | dupR | confR | xz_area | extent | gate |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 72 | submap_007 | accepted | needs_review | 0.066 | 0.539 | 0.112 | 0.178 | 0.710 | 45.6 | 8.4 | yes |
| 73 | submap_007 | accepted | needs_review | 0.094 | 0.442 | 0.105 | 0.128 | 0.767 | 81.5 | 10.1 | yes |
| 75 | submap_007 | needs_review | needs_review | 0.271 | 0.346 | 0.113 | 0.295 | 0.591 | 104.5 | 12.2 | yes |
| 76 | submap_007 | needs_review | needs_review | 0.347 | 0.395 | 0.089 | 0.338 | 0.573 | 109.4 | 12.4 | yes |
| 77 | submap_007 | needs_review | needs_review | 0.414 | 0.351 | 0.096 | 0.237 | 0.667 | 101.4 | 10.8 | yes |
| 148 | submap_014 | needs_review | needs_review | 1.213 | 0.020 | 0.354 | 0.000 | 0.646 | 36.2 | 7.1 | yes |
| 150 | submap_014 | needs_review | needs_review | 0.716 | 0.099 | 0.082 | 0.193 | 0.726 | 47.2 | 7.0 | yes |
| 151 | submap_015 | needs_review | needs_review | 0.122 | 0.212 | 0.127 | 0.110 | 0.762 | 64.9 | 9.0 | yes |
| 152 | submap_015 | needs_review | needs_review | 0.070 | 0.207 | 0.268 | 0.005 | 0.728 | 96.0 | 12.1 | yes |
| 164 | submap_016 | needs_review | needs_review | 0.180 | 0.067 | 0.176 | 0.005 | 0.820 | 201.4 | 17.6 | yes |
| 185 | submap_018 | needs_review | needs_review | 0.159 | 0.095 | 0.191 | 0.037 | 0.772 | 127.3 | 12.3 | yes |
| 186 | submap_018 | needs_review | needs_review | 0.164 | 0.340 | 0.103 | 0.142 | 0.755 | 119.1 | 12.3 | yes |
| 200 | submap_019 | accepted | needs_review | 0.053 | 0.508 | 0.057 | 0.169 | 0.773 | 28.6 | 8.4 | yes |
| 207 | submap_020 | needs_review | needs_review | 0.197 | 0.375 | 0.061 | 0.169 | 0.771 | 47.3 | 7.8 | yes |
| 273 | submap_027 | accepted | needs_review | 0.026 | 0.404 | 0.221 | 0.026 | 0.753 | 75.9 | 9.8 | yes |
| 289 | submap_028 | needs_review | needs_review | 0.367 | 0.139 | 0.144 | 0.096 | 0.760 | 156.4 | 14.5 | yes |

## Notes

- This is an offline diagnostic prototype. It does not modify the main mapping pipeline.
- `accepted_xz_area` is computed from `fusion_debug/accepted_points.ply`, so it reflects the points that the current fusion logic would write as new structure.
- The run does not save a per-step pre-fusion world. This prototype therefore uses accepted cloud morphology rather than nearest-surface features against the exact world-before-step.
