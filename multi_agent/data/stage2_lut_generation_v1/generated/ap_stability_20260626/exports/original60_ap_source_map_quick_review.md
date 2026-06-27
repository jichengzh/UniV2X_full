# Original60 AP Source Map Quick Review

| status | count |
| --- | ---: |
| exact_true_source_available | 4 |
| finetune_smoke_quarantined | 1 |
| no_claim_missing_source | 54 |
| weight_identity_transfer_available | 1 |

| label | width | status | AP70 | next_action |
| --- | --- | --- | ---: | --- |
| s0_024 | [24, 128, 256] | exact_true_source_available | 0.632495 | ap_row_available_from_phase_c_finetune |
| s0_040 | [40, 128, 256] | exact_true_source_available | 0.626659 | ap_row_available_from_phase_c_finetune |
| s0_056 | [56, 128, 256] | exact_true_source_available | 0.623711 | ap_row_available_from_phase_c_finetune |
| s1_048 | [64, 48, 256] | exact_true_source_available | 0.635616 | ap_row_available_from_phase_c_finetune |
| s1_064 | [64, 64, 256] | weight_identity_transfer_available | 0.636160 | can_claim_transfer_after_identity_gate |
| s1_080 | [64, 80, 256] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| s1_112 | [64, 112, 256] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| s2_096 | [64, 128, 96] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| s2_160 | [64, 128, 160] | finetune_smoke_quarantined |  | inspect_export_sanity_or_retrain_before_claim |
| s2_224 | [64, 128, 224] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_01 | [24, 64, 160] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_02 | [32, 96, 256] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_03 | [40, 128, 128] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_04 | [48, 48, 224] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_05 | [56, 80, 96] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_06 | [64, 112, 192] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_07 | [24, 48, 128] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_08 | [32, 80, 224] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_09 | [40, 112, 96] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_10 | [48, 32, 192] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_11 | [56, 64, 64] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_12 | [64, 96, 160] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_14 | [32, 64, 192] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_15 | [40, 96, 64] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_16 | [48, 128, 160] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_17 | [56, 48, 256] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_18 | [64, 80, 128] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_19 | [16, 112, 224] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_20 | [24, 32, 96] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_21 | [40, 80, 256] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_22 | [48, 112, 128] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_23 | [56, 32, 224] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_24 | [64, 64, 96] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_25 | [16, 96, 192] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| lhc_26 | [24, 128, 64] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_01 | [24, 64, 128] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_02 | [40, 64, 128] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_03 | [32, 48, 128] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_04 | [32, 80, 128] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_05 | [32, 64, 96] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_06 | [32, 64, 160] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_07 | [24, 48, 96] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_08 | [40, 80, 160] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_10 | [24, 32, 64] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_11 | [16, 48, 64] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_12 | [16, 32, 96] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_14 | [40, 96, 192] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_15 | [56, 96, 192] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_16 | [48, 80, 192] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_17 | [48, 112, 192] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_18 | [48, 96, 160] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_19 | [48, 96, 224] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_20 | [56, 112, 224] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_22 | [64, 80, 192] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_25 | [64, 96, 224] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_26 | [56, 80, 160] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_27 | [64, 112, 224] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_31 | [48, 112, 256] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_32 | [48, 128, 224] | no_claim_missing_source |  | needs_finetune_or_true_eval |
| frontier_33 | [40, 112, 224] | no_claim_missing_source |  | needs_finetune_or_true_eval |
