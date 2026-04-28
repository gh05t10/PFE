# Protocol: H1 - Patch Length Optimization

## Hypothesis
Reducing the `patch_len` from 16 to 8 will better capture high-frequency patterns and improve validation RMSE for the SlidePatchCrossAttn model.

## Rationale
The typical sequence length window is `L=96` (e.g. 4 days of hourly data). A patch size of 16 means patches cover 16 time steps. For rapidly changing local chlorophyll dynamics, 16 might smooth out the features too much. Reducing it to 8 allows the model's cross-attention mechanisms to map higher frequency temporal correlations, which might lower validation RMSE.

## Prediction
- Validation RMSE will decrease compared to the `patch-len=16` baseline.

## Method
1. Run baseline: `python run_train_slide.py --patch-len 16 --epochs 10`
   Log baseline proxy metric `val_rmse`.
2. Run experimental: `python run_train_slide.py --patch-len 8 --epochs 10`
   (Note: Epochs truncated to 10 for rapid inner loop turnover).
3. Compare `val_rmse` from both runs. 

## Measurable Outcome
`delta = val_rmse(patch-len=8) - val_rmse(patch-len=16)`. If `delta < 0`, hypothesis is CONFIRMATORY.
