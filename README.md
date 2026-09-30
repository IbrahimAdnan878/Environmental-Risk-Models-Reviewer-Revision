# Reviewer Revision: Complete Target-Specific Feature Lists

This folder contains supplementary material prepared in response to the reviewer request for the complete list of features used for each prediction target.

## File

`complete_feature_lists.csv`

The file reports the predictor sets used in the revised Random Forest, XGBoost and LSTM experiments for:

- drought_flag
- drought_severity
- precipitation_sum
- dust_event

For each model and target, the file provides:

- the model name;
- the prediction target;
- the number of retained predictor features;
- the complete list of included features;
- the leakage-sensitive features excluded from the predictor set.

The revised experiments use the same target-specific predictor sets across Random Forest, XGBoost and LSTM to support a fair comparison between models.

The retained feature counts are:

- drought_flag: 75 features
- drought_severity: 75 features
- precipitation_sum: 81 features
- dust_event: 70 features

Leakage-sensitive variables were removed before model training. For drought-related targets, variables directly involved in constructing the drought labels were excluded. For precipitation prediction, same-period precipitation and drought-derived variables were excluded. For dust-event prediction, current dust-proxy variables and current rolling proxies were excluded, while historical lagged variables were retained where appropriate.

The revised experiments use a chronological 70% training, 15% validation and 15% final test split. Validation data were used for model-selection decisions, while the final test set was reserved for independent performance reporting.
