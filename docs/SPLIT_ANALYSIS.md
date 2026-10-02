# Split Analysis

Historical Phase 2B candidate analysis. The final split is locked in [split_v1.yaml](../configs/split_v1.yaml): 621 train / 111 validation / 114 TEST. Candidate proposals below are not current selection instructions; no post-TEST changes are permitted.

Phase 2B builds the full manifest and proposes date-level splits only. No model training and no duplicated raster tile cache were created.

## Manifest Summary

- Tile size: `224 x 224`
- Tiling: non-overlapping windows on the B04 reference grid
- Minimum valid-pixel ratio: `0.2`
- Total usable tiles: `846`
- Source rasters are read on demand from file paths recorded in `results/tile_manifest.csv`.

## Per-Date Statistics

| date | valid_tiles | valid_pixels | class_0_pixels | class_1_pixels | class_2_pixels | class_3_pixels | algae_pixels | algae_fraction | tiles_with_low | tiles_with_mid | tiles_with_high |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-01-01 | 44 | 1174752 | 613341 | 544761 | 13338 | 3312 | 561411 | 0.478 | 44 | 36 | 17 |
| 2025-01-31 | 83 | 2935586 | 1139731 | 1772574 | 15347 | 7934 | 1795855 | 0.612 | 83 | 57 | 42 |
| 2025-03-12 | 38 | 1219390 | 1205125 | 1175 | 10523 | 2567 | 14265 | 0.012 | 23 | 38 | 17 |
| 2025-04-08 | 85 | 2948612 | 2826654 | 15259 | 95969 | 10730 | 121958 | 0.041 | 48 | 85 | 43 |
| 2025-04-09 | 86 | 2961449 | 2856348 | 15611 | 78541 | 10949 | 105101 | 0.035 | 47 | 73 | 43 |
| 2025-05-16 | 86 | 3126524 | 2682099 | 6735 | 425438 | 12252 | 444425 | 0.142 | 47 | 86 | 44 |
| 2025-05-18 | 86 | 3128333 | 2964646 | 7642 | 143693 | 12352 | 163687 | 0.052 | 46 | 86 | 44 |
| 2025-05-21 | 85 | 3250809 | 3173600 | 15101 | 47392 | 14716 | 77209 | 0.024 | 49 | 84 | 43 |
| 2025-06-20 | 67 | 1791726 | 1022963 | 609801 | 120747 | 38215 | 768763 | 0.429 | 66 | 67 | 65 |
| 2025-08-12 | 72 | 2159300 | 296616 | 184264 | 1573048 | 105372 | 1862684 | 0.863 | 72 | 71 | 72 |
| 2025-09-08 | 58 | 1618660 | 592216 | 939974 | 78434 | 8036 | 1026444 | 0.634 | 58 | 58 | 34 |
| 2025-09-21 | 56 | 1550433 | 1096199 | 444555 | 3778 | 5901 | 454234 | 0.293 | 55 | 44 | 28 |

## Candidate Date-Level Splits

Each candidate uses 8 train dates, 2 validation dates and 2 test dates. No date appears in more than one split. Validation and test sets each contain classes `1`, `2` and `3`.

| candidate | split | n_dates | dates | valid_tiles | valid_pixels | class_0 | class_1 | class_2 | class_3 | algae_fraction | has_low | has_mid | has_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| candidate_1 | train | 8 | 2025-01-31, 2025-03-12, 2025-04-09, 2025-05-18, 2025-05-21, 2025-06-20, 2025-08-12, 2025-09-21 | 573 | 18997026 | 13755228 | 3050723 | 1993069 | 198006 | 0.276 | True | True | True |
| candidate_1 | validation | 2 | 2025-01-01, 2025-05-16 | 130 | 4301276 | 3295440 | 551496 | 438776 | 15564 | 0.234 | True | True | True |
| candidate_1 | test | 2 | 2025-04-08, 2025-09-08 | 143 | 4567272 | 3418870 | 955233 | 174403 | 18766 | 0.251 | True | True | True |
| candidate_2 | train | 8 | 2025-01-31, 2025-03-12, 2025-04-09, 2025-05-18, 2025-05-21, 2025-06-20, 2025-08-12, 2025-09-21 | 573 | 18997026 | 13755228 | 3050723 | 1993069 | 198006 | 0.276 | True | True | True |
| candidate_2 | validation | 2 | 2025-04-08, 2025-09-08 | 143 | 4567272 | 3418870 | 955233 | 174403 | 18766 | 0.251 | True | True | True |
| candidate_2 | test | 2 | 2025-01-01, 2025-05-16 | 130 | 4301276 | 3295440 | 551496 | 438776 | 15564 | 0.234 | True | True | True |
| candidate_3 | train | 8 | 2025-01-31, 2025-03-12, 2025-04-08, 2025-05-18, 2025-05-21, 2025-06-20, 2025-08-12, 2025-09-21 | 572 | 18984189 | 13725534 | 3050371 | 2010497 | 197787 | 0.277 | True | True | True |
| candidate_3 | validation | 2 | 2025-01-01, 2025-05-16 | 130 | 4301276 | 3295440 | 551496 | 438776 | 15564 | 0.234 | True | True | True |
| candidate_3 | test | 2 | 2025-04-09, 2025-09-08 | 144 | 4580109 | 3448564 | 955585 | 156975 | 18985 | 0.247 | True | True | True |

## Trade-Offs

The candidates differ mainly in algae fraction and valid-pixel volume. Because bloom severity is date-dependent, the validation and test pairs cannot be perfectly balanced while preserving date-level leakage safety. Choose the final split only after deciding whether the priority is temporal order, class balance, or stress-testing on high-algae dates.
