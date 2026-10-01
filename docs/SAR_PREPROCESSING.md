# SAR Preprocessing Recovered from Day 2

Phase 3A inspected only the Day 2 lab notebook sections needed for SAR input handling.

Recovered preprocessing:

- Sentinel-1 channels are `VV` and `VH` in that order.
- SAR values are terrain-corrected Gamma0 backscatter in dB: `GAMMA0_TERRAIN` with `output_scale = db`.
- SAR nodata is `-9999`.
- VV/VH are aligned to the Sentinel-2 B04 grid with bilinear resampling.
- Masks are aligned to the B04 grid with nearest-neighbour resampling.
- The SAR-only DeepLab adapter receives a finite `[2, H, W]` tensor in `VV,VH` order.
- RGB reflectance scaling must not be applied to SAR.
- SAR CNN inputs are standardized with mean/std fit from valid training pixels only.
- Day 2 adapts the RGB DeepLab input stem to two channels by repeating the mean RGB stem weights and scaling by `3/2`.

Phase 3A fitted train-only SAR statistics on the fixed split:

- VV mean: `-22.144847`, std: `6.108022`
- VH mean: `-32.467809`, std: `6.304150`
- valid train pixels used per channel: `[21730003, 21730003]`

This project keeps the four-class target rather than Day 2's teaching binary target.
