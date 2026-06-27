# AP Source Inventory Summary

| source_kind | count |
| --- | ---: |
| model_fit_only | 1 |
| true_eval | 7 |
| true_import | 19 |
| weight_identity_transfer | 5 |

| label | width | source_kind | AP70 | claim_status | source_path |
| --- | --- | --- | ---: | --- | --- |
| base | [64, 128, 256] | true_import | 0.630864 | provisional_missing_digest | multi_agent/data/sources/stage_a_ap_real.csv |
| base_int8 | [64, 128, 256] | true_import | 0.622809 | provisional_missing_digest | multi_agent/data/sources/stage_a_ap_real.csv |
| trap25 | [48, 96, 192] | true_import | 0.590472 | provisional_missing_digest | multi_agent/data/sources/stage_a_ap_real.csv |
| trap25_int8 | [48, 96, 192] | true_import | 0.584088 | provisional_missing_digest | multi_agent/data/sources/stage_a_ap_real.csv |
| p50 | [32, 64, 128] | true_import | 0.564132 | provisional_missing_digest | multi_agent/data/sources/stage_a_ap_real.csv |
| p50_int8 | [32, 64, 128] | true_import | 0.554228 | provisional_missing_digest | multi_agent/data/sources/stage_a_ap_real.csv |
| p75 | [16, 32, 64] | true_import | 0.529988 | provisional_missing_digest | multi_agent/data/sources/stage_a_ap_real.csv |
| p75_int8 | [16, 32, 64] | true_import | 0.523636 | provisional_missing_digest | multi_agent/data/sources/stage_a_ap_real.csv |
| mix_b | [48, 64, 256] | true_eval | 0.636160 | provisional_missing_digest | results/ap70_depgraph_expansion.json |
| mix_d | [48, 128, 128] | true_eval | 0.636938 | provisional_missing_digest | results/ap70_depgraph_expansion.json |
| s1_64 | [64, 64, 256] | weight_identity_transfer | 0.636160 | claimable_weight_identity_transfer | results/ap70_depgraph_expansion.json |
| s2_128 | [64, 128, 128] | weight_identity_transfer | 0.636938 | claimable_weight_identity_transfer | results/ap70_depgraph_expansion.json |
| iso_s0 | [48, 128, 256] | true_eval | 0.629919 | provisional_missing_digest | results/b2_eval/b2_iso_s0_fp16.json |
| iso_s1 | [64, 96, 256] | true_eval | 0.633599 | provisional_missing_digest | results/b2_eval/b2_iso_s1_fp16.json |
| iso_s2 | [64, 128, 192] | true_eval | 0.633890 | provisional_missing_digest | results/b2_eval/b2_iso_s2_fp16.json |
| p50b2_136 | [32, 64, 136] | true_eval | 0.642530 | provisional_missing_digest | results/p0_2_136/p50b2_136_fp16.json |
| ap70_loglinear_model | [64, 128, 256] | model_fit_only |  | blocked_model_fit_only | results/ap70_model_pyramid.json |
| mixed | [32, 96, 192] | true_eval | 0.628755 | provisional_missing_digest | results/ap70_model_pyramid.json |
| pad64 | [64, 96, 192] | weight_identity_transfer | 0.590500 | claimable_weight_identity_transfer | results/ap70_model_pyramid.json |
| s1_64 | [64, 64, 256] | weight_identity_transfer | 0.636200 | claimable_weight_identity_transfer | results/ap70_model_pyramid.json |
| s2_128 | [64, 128, 128] | weight_identity_transfer | 0.636900 | claimable_weight_identity_transfer | results/ap70_model_pyramid.json |
| base | [64, 128, 256] | true_import | 0.630864 | provisional_missing_digest | multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl |
| p50 | [32, 64, 128] | true_import | 0.564132 | provisional_missing_digest | multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl |
| trap25 | [48, 96, 192] | true_import | 0.590472 | provisional_missing_digest | multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl |
| p75 | [16, 32, 64] | true_import | 0.529988 | provisional_missing_digest | multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl |
| mix_b | [48, 64, 256] | true_import | 0.636160 | provisional_missing_digest | multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl |
| mix_d | [48, 128, 128] | true_import | 0.636938 | provisional_missing_digest | multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl |
| iso_s0 | [48, 128, 256] | true_import | 0.629919 | provisional_missing_digest | multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl |
| iso_s1 | [64, 96, 256] | true_import | 0.633599 | provisional_missing_digest | multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl |
| iso_s2 | [64, 128, 192] | true_import | 0.633890 | provisional_missing_digest | multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl |
| mix_a | [32, 96, 192] | true_import | 0.628755 | provisional_missing_digest | multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl |
| p50b2_136 | [32, 64, 136] | true_import | 0.642530 | provisional_missing_digest | multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl |
