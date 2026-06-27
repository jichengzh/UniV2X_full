# Original60 AP/Latency/Energy 三指标总表

生成时间: 2026-06-27T02:10:06+00:00

## 汇总

| item | count |
|---|---:|
| total_rows | 60 |
| latency_measured_rows | 58 |
| energy_measured_rows | 56 |
| ap_with_ap70_rows | 5 |
| ap_true_eval_rows | 4 |
| ap_transfer_rows | 1 |
| ap_no_claim_rows | 55 |
| ap_quality_pending_repeat_rows | 4 |
| quarantined_rows | 5 |

说明: latency 单位为 ms; energy 单位为 joule/inference; AP 为空表示当前没有合法 true source 或 transfer source。

## 明细

| label | width | latency_default_ms | latency_tuned_ms | energy_j_per_inference | ap30 | ap50 | ap70 | latency_status | energy_status | ap_status | ap_quality_gate_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s0_024 | 24x128x256 | 39.499614 | 39.503066 | 4.954709 | 0.833141389 | 0.791044145 | 0.632495206 | measured | measured | claimable_true_eval | trend_anomaly_pending_repeat |
| s0_040 | 40x128x256 | 44.68875 | 44.704673 | 4.794238 | 0.822728829 | 0.78124512 | 0.626659483 | measured | measured | claimable_true_eval | first_seed_true_eval_pending_repeat |
| s0_056 | 56x128x256 | 51.351367 | 51.342031 | 6.196241 | 0.832072361 | 0.790658722 | 0.623711042 | measured | measured | claimable_true_eval | first_seed_true_eval_pending_repeat |
| s1_048 | 64x48x256 | 44.294917 | 44.452754 | 5.499634 | 0.822220053 | 0.782852439 | 0.635616211 | measured | measured | claimable_true_eval | trend_anomaly_pending_repeat |
| s1_064 | 64x64x256 | 46.437119 | 46.633232 | 4.817434 |  |  | 0.636159742 | measured | measured | claimable_weight_identity_transfer |  |
| s1_080 | 64x80x256 | 49.117573 | 49.101309 | 6.602432 |  |  |  | measured | measured | no_claim_missing_source |  |
| s1_112 | 64x112x256 | 54.167803 | 54.163374 | 8.87171 |  |  |  | measured | measured | no_claim_missing_source |  |
| s2_096 | 64x128x96 |  |  |  |  |  |  | quarantine_cuda_illegal_memory_access | not_run_latency_quarantine | no_claim_missing_source |  |
| s2_160 | 64x128x160 | 49.694395 | 49.77666 | 4.162604 |  |  |  | measured | measured | no_claim_missing_source | export_or_eval_quarantined |
| s2_224 | 64x128x224 | 54.299932 | 54.443613 | 6.093609 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_01 | 24x64x160 | 23.328262 | 23.326084 | 3.032124 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_02 | 32x96x256 | 39.970571 | 40.107417 | 1.774235 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_03 | 40x128x128 | 36.040906 | 36.060876 | 3.737698 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_04 | 48x48x224 | 37.45573 | 37.52106 | 3.944334 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_05 | 56x80x96 | 33.733826 | 29.18115 | 1.256554 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_06 | 64x112x192 | 49.889971 | 49.811182 | 6.56558 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_07 | 24x48x128 | 18.967836 | 12.952994 | 0 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_08 | 32x80x224 | 35.734133 | 24.642524 | 0.675306 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_09 | 40x112x96 | 32.096267 | 32.322161 | 3.121599 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_10 | 48x32x192 | 32.649292 | 32.633389 | 5.052279 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_11 | 56x64x64 | 29.004858 | 26.037461 | 1.215893 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_12 | 64x96x160 | 45.009268 | 37.253828 | 3.728277 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_14 | 32x64x192 | 30.742437 | 30.822122 | 2.499408 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_15 | 40x96x64 | 27.478977 | 24.559172 | 0.463489 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_16 | 48x128x160 | 44.690923 | 44.692739 | 5.514316 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_17 | 56x48x256 |  |  |  |  |  |  | quarantine_cuda_illegal_memory_access | not_run_latency_quarantine | no_claim_missing_source |  |
| lhc_18 | 64x80x128 | 40.416882 | 34.418411 | 3.050631 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_19 | 16x112x224 | 35.494946 | 35.515078 | 2.897031 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_20 | 24x32x96 | 14.708424 | 10.127507 | 0.362166 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_21 | 40x80x256 | 37.664858 | 37.757703 | 5.096023 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_22 | 48x112x128 | 40.436836 | 40.453059 | 4.407295 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_23 | 56x32x224 | 35.098279 | 31.911555 | 2.416399 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_24 | 64x64x96 | 35.879082 | 31.329392 | 1.141954 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_25 | 16x96x192 | 30.605505 | 30.631062 | 2.241105 |  |  |  | measured | measured | no_claim_missing_source |  |
| lhc_26 | 24x128x64 | 26.912761 | 23.985935 | 3.431983 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_01 | 24x64x128 | 21.064802 | 21.357256 | 0.424771 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_02 | 40x64x128 | 26.229487 | 26.283811 | 1.092068 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_03 | 32x48x128 | 24.120591 | 24.126619 | 0.158221 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_04 | 32x80x128 | 28.898157 | 28.934253 | 2.34175 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_05 | 32x64x96 | 24.351216 | 24.355515 | 0.525717 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_06 | 32x64x160 | 28.426504 | 20.674619 | 1.715429 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_07 | 24x48x96 | 17.12476 | 12.578511 | 0 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_08 | 40x80x160 | 31.230066 | 23.459915 | 0.045117 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_10 | 24x32x64 | 12.692048 | 9.826991 | 0 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_11 | 16x48x64 | 15.089955 | 15.16151 | 0 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_12 | 16x32x96 | 14.683405 | 10.106415 | 0.45561 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_14 | 40x96x192 | 35.860403 | 35.994951 | 3.356181 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_15 | 56x96x192 | 42.413501 | 42.422632 | 5.13018 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_16 | 48x80x192 | 39.963096 | 40.24512 | 2.946921 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_17 | 48x112x192 | 44.971597 | 47.607866 | 4.929561 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_18 | 48x96x160 | 40.016536 | 40.024951 | 5.132624 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_19 | 48x96x224 | 44.586284 | 44.635464 | 5.90908 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_20 | 56x112x224 | 47.398379 | 47.674604 | 6.774405 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_22 | 64x80x192 | 44.915273 | 45.002417 | 6.08687 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_25 | 64x96x224 | 49.619209 | 38.320701 |  |  |  |  | measured | quarantine_cuda_illegal_memory_access | no_claim_missing_source |  |
| frontier_26 | 56x80x160 | 37.777043 | 30.017478 |  |  |  |  | measured | quarantine_cuda_illegal_memory_access | no_claim_missing_source |  |
| frontier_27 | 64x112x224 | 52.25355 | 52.08062 | 7.767189 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_31 | 48x112x256 | 49.187661 | 49.42186 | 6.745455 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_32 | 48x128x224 | 49.30041 | 49.274531 | 8.185802 |  |  |  | measured | measured | no_claim_missing_source |  |
| frontier_33 | 40x112x224 | 40.759946 | 40.727156 | 6.111159 |  |  |  | measured | measured | no_claim_missing_source |  |
