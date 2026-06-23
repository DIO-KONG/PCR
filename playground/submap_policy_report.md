# Submap Policy Simulation

Run: `result/submap_walkforward/submap_327_align_only_se3_scale_gate_v2`

## Policy

```text
min_steps = 8
target_steps = 10
max_steps = 20
min_fused_for_rotation = 5
recent_window = 6
min_recent_fused = 2
max_steps_since_fused = 6
max_provisional_chain = 2
```

## Summary

- Steps: 315
- Pose rejected steps: 205
- Committed fusion steps: 88
- Scale quarantined steps: 22
- Steps using provisional target transform: 204
- Steps with provisional chain > 2: 169
- Simulated segments: 22
- Trusted rotations: 12
- Weak rotations: 0
- Detached/hold segments: 9
- Suspicious existing submap edges: 9

## Detached Or Hold Segments

| sim submap | steps | fused | pose rejected | scale gate | decision | reason |
|---|---:|---:|---:|---:|---|---|
| sim_submap_001 | 11-30 | 1 | 19 | 0 | detached_or_hold | max_steps_reached_without_reliable_fusion |
| sim_submap_005 | 65-84 | 4 | 11 | 5 | detached_or_hold | max_steps_reached_without_reliable_fusion |
| sim_submap_011 | 142-161 | 3 | 17 | 0 | detached_or_hold | max_steps_reached_without_reliable_fusion |
| sim_submap_012 | 162-181 | 0 | 18 | 2 | detached_or_hold | max_steps_reached_without_reliable_fusion |
| sim_submap_013 | 182-201 | 0 | 20 | 0 | detached_or_hold | max_steps_reached_without_reliable_fusion |
| sim_submap_014 | 202-221 | 0 | 20 | 0 | detached_or_hold | max_steps_reached_without_reliable_fusion |
| sim_submap_017 | 245-264 | 0 | 19 | 1 | detached_or_hold | max_steps_reached_without_reliable_fusion |
| sim_submap_018 | 265-284 | 0 | 20 | 0 | detached_or_hold | max_steps_reached_without_reliable_fusion |
| sim_submap_019 | 285-304 | 2 | 10 | 8 | detached_or_hold | max_steps_reached_without_reliable_fusion |

## Existing Suspicious Submap Edges

| edge | anchor | translation m | yaw-like deg |
|---|---|---:|---:|
| submap_001 -> submap_000 | anchor=batch_007_window_014-018 | 0.757 | 173.6 |
| submap_003 -> submap_002 | anchor=batch_027_window_034-038 | 1.732 | -146.7 |
| submap_004 -> submap_003 | anchor=batch_038_window_045-049 | 1.389 | 99.7 |
| submap_005 -> submap_004 | anchor=batch_049_window_056-060 | 1.400 | 168.6 |
| submap_006 -> submap_005 | anchor=batch_061_window_068-072 | 0.898 | -153.8 |
| submap_014 -> submap_013 | anchor=batch_141_window_148-152 | 4.133 | -81.0 |
| submap_015 -> submap_014 | anchor=batch_150_window_157-161 | 0.316 | 165.5 |
| submap_023 -> submap_022 | anchor=batch_230_window_237-241 | 5.969 | 111.4 |
| submap_031 -> submap_030 | anchor=batch_308_window_315-319 | 8.465 | -115.2 |

## Long Provisional Target Chain

| step | target | chain | pose | fusion | reasons |
|---:|---|---:|---|---|---|
| 10 | batch_010_window_017-021 | 3 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 11 | batch_011_window_018-022 | 4 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 12 | batch_012_window_019-023 | 5 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 13 | batch_013_window_020-024 | 6 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 14 | batch_014_window_021-025 | 7 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 15 | batch_015_window_022-026 | 8 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 16 | batch_016_window_023-027 | 9 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 17 | batch_017_window_024-028 | 10 | rejected | rejected | shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 18 | batch_018_window_025-029 | 11 | rejected | rejected | shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 19 | batch_019_window_026-030 | 12 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 20 | batch_020_window_027-031 | 13 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 21 | batch_021_window_028-032 | 14 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 22 | batch_022_window_029-033 | 15 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 23 | batch_023_window_030-034 | 16 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 24 | batch_024_window_031-035 | 17 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 25 | batch_025_window_032-036 | 18 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 26 | batch_026_window_033-037 | 19 | needs_review | needs_review | icp_fitness_too_low; fusion_conflict_ratio_too_high |
| 30 | batch_030_window_037-041 | 3 | rejected | rejected | icp_translation_delta_too_large; icp_rejected_by_boundary |
| 31 | batch_031_window_038-042 | 4 | rejected | rejected | icp_translation_delta_too_large; icp_rejected_by_boundary |
| 32 | batch_032_window_039-043 | 5 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 33 | batch_033_window_040-044 | 6 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 34 | batch_034_window_041-045 | 7 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 35 | batch_035_window_042-046 | 8 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 36 | batch_036_window_043-047 | 9 | rejected | rejected | icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 37 | batch_037_window_044-048 | 10 | needs_review | needs_review | icp_fitness_too_low; fusion_conflict_ratio_too_high |
| 41 | batch_041_window_048-052 | 3 | needs_review | needs_review | icp_fitness_too_low; fusion_conflict_ratio_too_high |
| 78 | batch_078_window_085-089 | 3 | needs_review | rejected | shared_p90_error_too_high; icp_fitness_too_low; fusion_conflict_ratio_too_high; fusion_scale_shadow_quarantined |
| 82 | batch_082_window_089-093 | 3 | rejected | rejected | shared_p90_error_too_high; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 83 | batch_083_window_090-094 | 4 | rejected | rejected | icp_translation_delta_too_large; icp_rejected_by_boundary |
| 84 | batch_084_window_091-095 | 5 | rejected | rejected | shared_p90_error_too_high; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 85 | batch_085_window_092-096 | 6 | rejected | rejected | icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 86 | batch_086_window_093-097 | 7 | rejected | rejected | shared_p90_error_too_high; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 87 | batch_087_window_094-098 | 8 | rejected | rejected | icp_translation_delta_too_large; icp_rejected_by_boundary |
| 88 | batch_088_window_095-099 | 9 | rejected | rejected | icp_translation_delta_too_large; icp_rejected_by_boundary |
| 89 | batch_089_window_096-100 | 10 | rejected | rejected | shared_p90_error_too_high; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 90 | batch_090_window_097-101 | 11 | needs_review | needs_review | shared_median_error_too_high; shared_p90_error_too_high; fusion_conflict_ratio_too_high |
| 117 | batch_117_window_124-128 | 3 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 118 | batch_118_window_125-129 | 4 | needs_review | needs_review | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; fusion_conflict_ratio_too_high |
| 126 | batch_126_window_133-137 | 3 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 127 | batch_127_window_134-138 | 4 | needs_review | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; fusion_conflict_ratio_too_high; fusion_scale_shadow_quarantined |
| 131 | batch_131_window_138-142 | 3 | rejected | rejected | shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 132 | batch_132_window_139-143 | 4 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 133 | batch_133_window_140-144 | 5 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 134 | batch_134_window_141-145 | 6 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 135 | batch_135_window_142-146 | 7 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 136 | batch_136_window_143-147 | 8 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 137 | batch_137_window_144-148 | 9 | needs_review | needs_review | shared_p90_error_too_high; icp_fitness_too_low; fusion_conflict_ratio_too_high |
| 147 | batch_147_window_154-158 | 3 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 148 | batch_148_window_155-159 | 4 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 149 | batch_149_window_156-160 | 5 | needs_review | needs_review | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; fusion_conflict_ratio_too_high |
| 153 | batch_153_window_160-164 | 3 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 154 | batch_154_window_161-165 | 4 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 155 | batch_155_window_162-166 | 5 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 156 | batch_156_window_163-167 | 6 | rejected | rejected | shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 157 | batch_157_window_164-168 | 7 | rejected | rejected | shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 158 | batch_158_window_165-169 | 8 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 159 | batch_159_window_166-170 | 9 | rejected | rejected | shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 160 | batch_160_window_167-171 | 10 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 161 | batch_161_window_168-172 | 11 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 162 | batch_162_window_169-173 | 12 | rejected | rejected | shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 163 | batch_163_window_170-174 | 13 | rejected | rejected | shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rejected_by_boundary |
| 164 | batch_164_window_171-175 | 14 | needs_review | rejected | shared_p90_error_too_high; icp_fitness_too_low; fusion_conflict_ratio_too_high; fusion_scale_shadow_quarantined |
| 169 | batch_169_window_176-180 | 3 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 170 | batch_170_window_177-181 | 4 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 171 | batch_171_window_178-182 | 5 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 172 | batch_172_window_179-183 | 6 | rejected | rejected | shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 173 | batch_173_window_180-184 | 7 | rejected | rejected | shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 174 | batch_174_window_181-185 | 8 | rejected | rejected | shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 175 | batch_175_window_182-186 | 9 | rejected | rejected | shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 176 | batch_176_window_183-187 | 10 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 177 | batch_177_window_184-188 | 11 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 178 | batch_178_window_185-189 | 12 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 179 | batch_179_window_186-190 | 13 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 180 | batch_180_window_187-191 | 14 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 181 | batch_181_window_188-192 | 15 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 182 | batch_182_window_189-193 | 16 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 183 | batch_183_window_190-194 | 17 | rejected | rejected | shared_median_error_too_high; shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 184 | batch_184_window_191-195 | 18 | rejected | rejected | shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 185 | batch_185_window_192-196 | 19 | rejected | rejected | shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |
| 186 | batch_186_window_193-197 | 20 | rejected | rejected | shared_p90_error_too_high; icp_fitness_too_low; icp_translation_delta_too_large; icp_rotation_delta_too_large; icp_rejected_by_boundary |

## Interpretation

- A detached/hold segment means fixed-size submap rotation would be unsafe because the local map did not grow enough.
- A long provisional target chain means ICP initialization is being propagated through rejected poses rather than through fused map anchors.
- Suspicious existing edges are not proof of visual failure, but they are high-priority candidates for submap edge sanity gates.
