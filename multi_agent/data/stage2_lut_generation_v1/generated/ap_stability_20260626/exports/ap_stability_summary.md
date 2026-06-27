# AP Stability Summary

Replay gate: `pass` (4/4 pass)

| label | width | AP30 | AP50 | AP70 | source_kind | claim_status | gpu | finetune_runs | epoches | ckpt_digest |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| base | [64, 128, 256] | 0.833159 | 0.791041 | 0.630864 | true_import | provisional_missing_digest |  |  |  | unknown |
| p50 | [32, 64, 128] | 0.818254 | 0.764428 | 0.564132 | true_import | provisional_missing_digest |  |  |  | unknown |
| p75 | [16, 32, 64] | 0.815909 | 0.756669 | 0.529988 | true_import | provisional_missing_digest |  |  |  | unknown |
| trap25 | [48, 96, 192] | 0.824434 | 0.776939 | 0.590472 | true_import | provisional_missing_digest |  |  |  | unknown |
| mix_b | [48, 64, 256] | 0.818581 | 0.780496 | 0.636160 | true_eval | provisional_missing_digest |  |  |  | unknown |
| mix_d | [48, 128, 128] | 0.825772 | 0.785835 | 0.636938 | true_eval | provisional_missing_digest |  |  |  | unknown |
| iso_s0 | [48, 128, 256] | 0.823978 | 0.784323 | 0.629919 | true_eval | provisional_missing_digest |  |  |  | unknown |
| iso_s1 | [64, 96, 256] | 0.830391 | 0.789614 | 0.633599 | true_eval | provisional_missing_digest |  |  |  | unknown |
| iso_s2 | [64, 128, 192] | 0.821267 | 0.783871 | 0.633890 | true_eval | provisional_missing_digest |  |  |  | unknown |
| p50b2_136 | [32, 64, 136] | 0.827085 | 0.787692 | 0.642530 | true_eval | provisional_missing_digest |  |  |  | unknown |
| s0_040 | [40, 128, 256] | 0.822729 | 0.781245 | 0.626659 | true_eval | claimable_true_eval | 1 | 1 | 31 | 6f5959572dd7525565365ffd642605237ccef5ec189f9bcdc27ebc0cbbffccf6 |
| s0_056 | [56, 128, 256] | 0.832072 | 0.790659 | 0.623711 | true_eval | claimable_true_eval | 2 | 1 | 31 | 1599b1b58eca803b3216df915dfab34d4a5283f688ae58efd62dd25e474691c8 |
| s1_048 | [64, 48, 256] | 0.822220 | 0.782852 | 0.635616 | true_eval | claimable_true_eval | 3 | 1 | 31 | 6ea24f149d57e61e3671a92bd100ce8c8084765a05b494561efa32c5bb36b19d |
| s0_024 | [24, 128, 256] | 0.833141 | 0.791044 | 0.632495 | true_eval | claimable_true_eval | 0 | 1 | 31 | 74bdf34458d0972f519b57f6862a936a57e4ebeba7087e34b97af1dd9b6d7009 |

## Phase C Quarantine

| label | status | reason | raw_artifact |
| --- | --- | --- | --- |
| s2_160 | failed | JobFailure:onnx_export_failed_rc_1 | /home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/raw/s2_160 |
