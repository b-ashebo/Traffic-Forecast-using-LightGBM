# Traffic Speed & TTI Forecasting (LightGBM)

LightGBM regressors that forecast **traffic speed (mph)** and the **Travel Time Index (TTI)** from 15-minute interval traffic data.

## Highlights

- **Feature engineering** — temporal features (hour, day of week, month, weekend flag) plus cyclical sine/cosine encodings; label-encoded categoricals.
- **Leakage-safe splitting** — chronological 70/15/15 train/val/test split with early stopping on the validation set.
- **Validation** — 5-fold `TimeSeriesSplit` cross-validation, reported with MAE, RMSE, MAPE, and R².


**Dependencies:** `lightgbm`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `matplotlib`.
