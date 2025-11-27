Conclusions: SOH & SOP Modeling

Model Accuracy
The Random Forest models achieved exceptional accuracy using only charging data:

| Metric | SOH Model | SOP Model |
| :--- | :---: | :---: |
| **MAPE** (Mean Absolute Percentage Error) | **~1.70%** | **~0.31%** |
| **MAE** (Mean Absolute Error) | **0.00905** | **0.00011** |
| **RMSE** (Root Mean Square Error) | **0.0189** | **0.0005** |

## Key Insights
1.  Feasibility: It is highly feasible to grade battery health (SOH) and power capability (SOP) without performing a time-consuming discharge test.
2.  SOP Prediction: The SOP model is extremely accurate (0.3% error). This suggests that the internal resistance (which limits power) has a very strong signature during the charging phase (likely visible in the CC-CV transition curve).
3.  Rated Power Correction: The model correctly utilized the global rated power of **26W** (observed in OEM 2) rather than the local max of OEM 1 (11W), ensuring the SOP metric is standardized across the entire feedstock.
