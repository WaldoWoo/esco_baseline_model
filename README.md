# esco_baseline_model

This repository contains a four-part pipeline for developing a baseline energy usage model using both tree-based ensemble methods and deep learning techniques. It is designed to support guaranteed energy savings projects typical of the Energy Services Company (ESCO) industry by establishing accurate and defensible energy usage baselines.

## Repository Structure

### `01_esco_prepare.ipynb`  
Prepares and feature-engineers the raw energy usage data for modeling. Includes:
- Timestamp alignment and resampling
- Weather and calendar feature engineering
- Lag features and derived metrics
- Data splitting and serialization

### `02_esco_train.ipynb`  
Trains a supervised regression ensemble using:
- LightGBM and XGBoost models
- Cross-validation and feature importance
- Baseline performance metrics (R², RMSE, MAPE)

### `03_esco_predict.ipynb`  
Generates predictions on validation and test datasets using the trained ensemble model. Includes:
- Model loading and prediction
- Evaluation metrics (RMSLE, MAPE)
- Time-series plots of actual vs. predicted usage

### `04_esco_predict_RNN.ipynb`  
Implements an LSTM-based Recurrent Neural Network (RNN) to model temporal energy usage patterns. Includes:
- Data reshaping for sequence modeling
- Manual hyperparameter tuning
- Dropout regularization and early stopping
- RMSLE-based evaluation and test set performance

Performance of the RNN model was comparable to that of the ensemble methods, with similar RMSLE and MAPE values. While the RNN was better at capturing temporal dynamics, the tree-based models provided clearer insight into feature importance.

## Objective

This project simulates a real-world ESCO workflow where accurate baseline modeling is critical for validating guaranteed savings. The models are applied to hourly facility energy usage data, with the goal of creating a reliable and interpretable baseline that supports measurement and verification (M&V) activities.

## Environment

These notebooks are currently designed for use in Google Colab and rely on:
- `pandas`, `numpy`
- `scikit-learn`, `matplotlib`
- `xgboost`, `lightgbm`
- `tensorflow` / `keras`

Before production deployment or public GitHub sharing, code should be:
- Modularized into `.py` scripts or cleaned `.ipynb` notebooks
- Refactored for local or cloud deployment
- Wrapped with CLI or API interfaces for reproducibility, if needed

## Evaluation Metrics

- RMSLE (Root Mean Squared Logarithmic Error)
- MAPE (Mean Absolute Percentage Error)
- RMSE and MAE for ensemble methods

## Notes

- Designed for baseline modeling in M&V contexts (e.g., IPMVP Option C)
- RNN models are useful for capturing temporal dependencies
- Tree-based models offer transparent feature importance and fast training

## Disclaimer

This repository is a demonstration of modeling techniques and is not intended for direct use in contractual M&V without further validation, normalization, and review in accordance with ESCO industry standards.
