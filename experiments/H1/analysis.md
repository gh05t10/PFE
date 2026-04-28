# Analysis: H1 - Patch Length Optimization

## Outcomes
- **Baseline (`patch_len=16`)**: val_RMSE = 1.9415
- **Experimental (`patch_len=8`)**: val_RMSE = 1.9440
- **Delta**: +0.0025 (Worse)

## Conclusion
Hypothesis is REFUTED.
Reducing the patch length from 16 to 8 slightly increased the validation RMSE. This suggests that a smaller patch size does not meaningfully capture beneficial high-frequency patterns, or it inadvertently increases the attention complexity without offering proportional gains. The model may rely more on mid-range temporal structures which are adequately captured at `patch_len=16`.
