# Environmental Risk Models — Reviewer Revision

This repository contains the revised machine-learning programs and supplementary feature documentation prepared in response to the reviewers' comments.

## Contents

The repository includes:

- revised Random Forest prediction program;
- revised XGBoost prediction program;
- revised LSTM prediction program;
- revised final prediction combiner;
- complete target-specific feature lists.

## Complete Feature Lists

The file `complete_feature_lists.csv` reports the predictor sets used in the revised Random Forest, XGBoost and LSTM experiments for:

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

## Revised Evaluation Strategy

The revised experiments use a chronological 70% training, 15% validation and 15% final test split, without random shuffling.

The validation subset was used for hyperparameter assessment, decision-threshold selection, early stopping where applicable, and model-selection decisions. The final test subset was used for final performance reporting.

Class imbalance was handled consistently across the classification models:

- Random Forest used balanced class weighting;
- XGBoost used balanced sample weights;
- LSTM used class weights derived from the training subset.

## Final Prediction Combiner

The final prediction combiner selects the best-performing model for each prediction target using validation-set performance.

- classification targets: highest validation F1-score;
- precipitation regression: lowest validation RMSE.

Dust-event model selection is performed separately for each city, whereas drought flag, drought severity and precipitation sum are selected globally across the five study cities.

The final test results are reported separately and are not used to determine the winning model.
