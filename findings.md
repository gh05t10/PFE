# Research Findings

## Research Question
How do variations in SlidePatchCrossAttn hyperparameters, specifically patch size, affect validation RMSE for shallow chlorophyll prediction?

## Current Understanding
We are evaluating `SlidePatchCrossAttn` models for time series forecasting. The default baseline code uses `patch_len=16`. We explored optimizing the temporal resolution of patches down to 8. We found that smaller patches do not improve prediction performance.

## Key Results
- **H1 (Patch size 8)**: Did not improve validation RMSE (1.9440 vs baseline 1.9415).

## Patterns and Insights
Initial architectural tuning of the primary attention dimensions suggests that the target features are macroscopic enough that a patch length of 16 adequately covers them, and adding temporal resolution does not isolate better gradients.

## Lessons and Constraints
- Decreasing `patch_len` alone might inject unnecessary context noise or overfit high-frequency dynamics without delivering proportional validation gains.

## Open Questions
- If reducing patch size isn't the solution, would increasing model heads or changing the loss function (e.g., using TDAlign exclusively) yield better learning signals?

## Optimization Trajectory
- Baseline (`patch_len=16`): **1.9415** RMSE
- H1 (`patch_len=8`): **1.9440** RMSE
*(Current best: 1.9415)*
