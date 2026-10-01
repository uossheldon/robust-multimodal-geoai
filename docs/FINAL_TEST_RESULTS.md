# Final Held-Out Test Results

Phase 7: first and final held-out test evaluation. No post-test tuning was performed.

## Audit

Test dates: 2025-09-08, 2025-09-21
Test tiles: 114
TEST results were not used to select checkpoints or alter model settings.

## Clean held-out test results

| model | condition | n | mean_iou_mean | mean_iou_std | macro_dice_mean | macro_dice_std | iou_background_mean | iou_background_std | iou_low_mean | iou_low_std | iou_mid_mean | iou_mid_std | iou_high_mean | iou_high_std | binary_algae_iou_mean | binary_algae_iou_std | binary_algae_dice_mean | binary_algae_dice_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_fusion | clean | 1 | 0.1658 |  | 0.2700 |  | 0.1653 |  | 0.3287 |  | 0.0265 |  | 0.1428 |  | 0.4426 |  | 0.6136 |  |
| modality_dropout | clean | 3 | 0.1826 | 0.0123 | 0.2987 | 0.0182 | 0.2764 | 0.0120 | 0.2402 | 0.0365 | 0.0475 | 0.0087 | 0.1662 | 0.0160 | 0.4024 | 0.0410 | 0.5731 | 0.0421 |
| occlusion_trained | clean | 3 | 0.1882 | 0.0293 | 0.3033 | 0.0380 | 0.2542 | 0.0721 | 0.3074 | 0.0403 | 0.0504 | 0.0172 | 0.1406 | 0.0042 | 0.4446 | 0.0299 | 0.6151 | 0.0289 |
| s1_weighted | clean | 1 | 0.1756 |  | 0.2799 |  | 0.2426 |  | 0.3251 |  | 0.0028 |  | 0.1318 |  | 0.3424 |  | 0.5101 |  |
| s2_weighted | clean | 1 | 0.1698 |  | 0.2541 |  | 0.4824 |  | 0.1054 |  | 0.0490 |  | 0.0425 |  | 0.4547 |  | 0.6252 |  |
| terramind_frozen | clean | 3 | 0.1718 | 0.0177 | 0.2843 | 0.0235 | 0.2099 | 0.0166 | 0.2557 | 0.0604 | 0.0449 | 0.0094 | 0.1766 | 0.0275 | 0.5176 | 0.0029 | 0.6821 | 0.0025 |

## Robust-model test curves summary

| model | condition | n | mean_iou_mean | mean_iou_std | macro_dice_mean | macro_dice_std | iou_background_mean | iou_background_std | iou_low_mean | iou_low_std | iou_mid_mean | iou_mid_std | iou_high_mean | iou_high_std | binary_algae_iou_mean | binary_algae_iou_std | binary_algae_dice_mean | binary_algae_dice_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| modality_dropout | clean | 3 | 0.1826 | 0.0123 | 0.2987 | 0.0182 | 0.2764 | 0.0120 | 0.2402 | 0.0365 | 0.0475 | 0.0087 | 0.1662 | 0.0160 | 0.4024 | 0.0410 | 0.5731 | 0.0421 |
| modality_dropout | missing_s1 | 3 | 0.1141 | 0.0166 | 0.1963 | 0.0218 | 0.1676 | 0.0405 | 0.1977 | 0.0801 | 0.0402 | 0.0032 | 0.0510 | 0.0102 | 0.4497 | 0.0198 | 0.6203 | 0.0190 |
| modality_dropout | missing_s2 | 3 | 0.1690 | 0.0176 | 0.2614 | 0.0278 | 0.4110 | 0.0882 | 0.0993 | 0.1048 | 0.0282 | 0.0067 | 0.1376 | 0.0056 | 0.1761 | 0.1088 | 0.2891 | 0.1671 |
| modality_dropout | occlusion_10 | 9 | 0.1840 | 0.0037 | 0.2648 | 0.0058 | 0.5321 | 0.0027 | 0.0060 | 0.0082 | 0.0517 | 0.0098 | 0.1461 | 0.0149 | 0.0650 | 0.0414 | 0.1196 | 0.0709 |
| modality_dropout | occlusion_30 | 9 | 0.1757 | 0.0040 | 0.2514 | 0.0065 | 0.5308 | 0.0046 | 0.0053 | 0.0062 | 0.0410 | 0.0093 | 0.1257 | 0.0114 | 0.0603 | 0.0459 | 0.1108 | 0.0795 |
| modality_dropout | occlusion_50 | 9 | 0.1684 | 0.0038 | 0.2383 | 0.0057 | 0.5318 | 0.0062 | 0.0013 | 0.0017 | 0.0269 | 0.0086 | 0.1137 | 0.0088 | 0.0654 | 0.0701 | 0.1159 | 0.1178 |
| modality_dropout | occlusion_70 | 9 | 0.1630 | 0.0055 | 0.2276 | 0.0092 | 0.5336 | 0.0013 | 0.0000 | 0.0000 | 0.0085 | 0.0065 | 0.1100 | 0.0180 | 0.0168 | 0.0119 | 0.0327 | 0.0227 |
| occlusion_trained | clean | 3 | 0.1882 | 0.0293 | 0.3033 | 0.0380 | 0.2542 | 0.0721 | 0.3074 | 0.0403 | 0.0504 | 0.0172 | 0.1406 | 0.0042 | 0.4446 | 0.0299 | 0.6151 | 0.0289 |
| occlusion_trained | missing_s1 | 3 | 0.1204 | 0.0264 | 0.2050 | 0.0423 | 0.1395 | 0.0823 | 0.2393 | 0.0116 | 0.0460 | 0.0069 | 0.0568 | 0.0054 | 0.4461 | 0.0193 | 0.6168 | 0.0185 |
| occlusion_trained | missing_s2 | 3 | 0.1993 | 0.0164 | 0.3116 | 0.0266 | 0.3171 | 0.1438 | 0.2919 | 0.1290 | 0.0466 | 0.0276 | 0.1418 | 0.0152 | 0.3152 | 0.1271 | 0.4693 | 0.1564 |
| occlusion_trained | occlusion_10 | 9 | 0.2173 | 0.0056 | 0.3330 | 0.0064 | 0.4380 | 0.0773 | 0.2377 | 0.0812 | 0.0608 | 0.0077 | 0.1326 | 0.0181 | 0.3561 | 0.0796 | 0.5206 | 0.0886 |
| occlusion_trained | occlusion_30 | 9 | 0.2233 | 0.0131 | 0.3450 | 0.0167 | 0.3126 | 0.0671 | 0.3852 | 0.0324 | 0.0637 | 0.0107 | 0.1317 | 0.0185 | 0.4355 | 0.0279 | 0.6063 | 0.0273 |
| occlusion_trained | occlusion_50 | 9 | 0.2084 | 0.0191 | 0.3250 | 0.0253 | 0.2792 | 0.1088 | 0.3675 | 0.0379 | 0.0585 | 0.0079 | 0.1284 | 0.0162 | 0.4288 | 0.0224 | 0.5999 | 0.0219 |
| occlusion_trained | occlusion_70 | 9 | 0.2122 | 0.0115 | 0.3279 | 0.0157 | 0.3742 | 0.0916 | 0.2971 | 0.0890 | 0.0507 | 0.0057 | 0.1270 | 0.0155 | 0.3782 | 0.0447 | 0.5475 | 0.0467 |

## Per-date behavior

| model | condition | seed | occlusion_fraction | date | valid_pixels | mean_iou | macro_dice | iou_background | iou_low | iou_mid | iou_high | binary_algae_iou | binary_algae_dice | run_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s2_weighted | clean |  |  | 2025-09-08 | 1754102 | 0.0914 | 0.1646 | 0.1539 | 0.1098 | 0.0568 | 0.0451 | 0.5416 | 0.7027 | single_run |
| s2_weighted | clean |  |  | 2025-09-21 | 1720182 | 0.2028 | 0.2667 | 0.6683 | 0.0965 | 0.0056 | 0.0410 | 0.2491 | 0.3988 | single_run |
| s1_weighted | clean |  |  | 2025-09-08 | 1618660 | 0.1846 | 0.2875 | 0.2102 | 0.3944 | 0.0027 | 0.1311 | 0.4128 | 0.5844 | single_run |
| s1_weighted | clean |  |  | 2025-09-21 | 1550433 | 0.1639 | 0.2666 | 0.2699 | 0.2482 | 0.0047 | 0.1327 | 0.2614 | 0.4145 | single_run |
| early_fusion | clean |  |  | 2025-09-08 | 1618660 | 0.1656 | 0.2612 | 0.1036 | 0.4062 | 0.0402 | 0.1123 | 0.5843 | 0.7376 | single_run |
| early_fusion | clean |  |  | 2025-09-21 | 1550433 | 0.1540 | 0.2559 | 0.2023 | 0.2315 | 0.0047 | 0.1776 | 0.2754 | 0.4319 | single_run |
| modality_dropout | clean | 42.0000 |  | 2025-09-08 | 1618660 | 0.1760 | 0.2916 | 0.1999 | 0.2784 | 0.0613 | 0.1644 | 0.4481 | 0.6188 | seeded |
| modality_dropout | clean | 42.0000 |  | 2025-09-21 | 1550433 | 0.1883 | 0.2995 | 0.3258 | 0.2354 | 0.0057 | 0.1863 | 0.2449 | 0.3935 | seeded |
| modality_dropout | clean | 7.0000 |  | 2025-09-08 | 1618660 | 0.1250 | 0.2183 | 0.1361 | 0.1834 | 0.0415 | 0.1391 | 0.5767 | 0.7315 | seeded |
| modality_dropout | clean | 7.0000 |  | 2025-09-21 | 1550433 | 0.1899 | 0.2978 | 0.3786 | 0.2150 | 0.0061 | 0.1601 | 0.2411 | 0.3885 | seeded |
| modality_dropout | clean | 123.0000 |  | 2025-09-08 | 1618660 | 0.1626 | 0.2711 | 0.1374 | 0.2884 | 0.0576 | 0.1671 | 0.5344 | 0.6966 | seeded |
| modality_dropout | clean | 123.0000 |  | 2025-09-21 | 1550433 | 0.1945 | 0.3064 | 0.3507 | 0.2350 | 0.0052 | 0.1872 | 0.2414 | 0.3889 | seeded |
| occlusion_trained | clean | 42.0000 |  | 2025-09-08 | 1618660 | 0.1818 | 0.2954 | 0.1870 | 0.3452 | 0.0748 | 0.1199 | 0.5272 | 0.6904 | seeded |
| occlusion_trained | clean | 42.0000 |  | 2025-09-21 | 1550433 | 0.1887 | 0.2994 | 0.3326 | 0.2498 | 0.0085 | 0.1639 | 0.2626 | 0.4160 | seeded |
| occlusion_trained | clean | 7.0000 |  | 2025-09-08 | 1618660 | 0.1634 | 0.2628 | 0.0934 | 0.3732 | 0.0524 | 0.1345 | 0.5683 | 0.7247 | seeded |
| occlusion_trained | clean | 7.0000 |  | 2025-09-21 | 1550433 | 0.2321 | 0.3466 | 0.4467 | 0.3176 | 0.0099 | 0.1540 | 0.3176 | 0.4821 | seeded |
| occlusion_trained | clean | 123.0000 |  | 2025-09-08 | 1618660 | 0.1313 | 0.2218 | 0.0765 | 0.2754 | 0.0431 | 0.1304 | 0.6012 | 0.7509 | seeded |
| occlusion_trained | clean | 123.0000 |  | 2025-09-21 | 1550433 | 0.1671 | 0.2726 | 0.2291 | 0.2652 | 0.0032 | 0.1710 | 0.2688 | 0.4237 | seeded |
| terramind_frozen | clean | 42.0000 |  | 2025-09-08 | 1618660 | 0.1465 | 0.2402 | 0.0402 | 0.3095 | 0.0539 | 0.1823 | 0.6422 | 0.7821 | seeded |
| terramind_frozen | clean | 42.0000 |  | 2025-09-21 | 1550433 | 0.2275 | 0.3607 | 0.2713 | 0.3412 | 0.0847 | 0.2129 | 0.3511 | 0.5197 | seeded |
| terramind_frozen | clean | 7.0000 |  | 2025-09-08 | 1618660 | 0.0701 | 0.1293 | 0.0318 | 0.0907 | 0.0464 | 0.1114 | 0.6404 | 0.7808 | seeded |
| terramind_frozen | clean | 7.0000 |  | 2025-09-21 | 1550433 | 0.2279 | 0.3522 | 0.3172 | 0.3382 | 0.0201 | 0.2363 | 0.3583 | 0.5276 | seeded |
| terramind_frozen | clean | 123.0000 |  | 2025-09-08 | 1618660 | 0.0730 | 0.1306 | 0.0169 | 0.0571 | 0.0449 | 0.1732 | 0.6377 | 0.7788 | seeded |
| terramind_frozen | clean | 123.0000 |  | 2025-09-21 | 1550433 | 0.2450 | 0.3657 | 0.3274 | 0.4263 | 0.0040 | 0.2223 | 0.3563 | 0.5254 | seeded |

## Generalization notes

- All models drop substantially relative to validation macro mIoU, indicating temporal generalization difficulty on September dates.
- TerraMind has the highest binary algae Dice among clean models, while occlusion-trained fusion has the best clean macro mIoU among robust DeepLab variants.
- Mid and high algae classes remain the hardest classes across models.
