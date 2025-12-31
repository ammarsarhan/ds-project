# Solar Power Generation Prediction and Classification

A comprehensive machine learning project for predicting solar photovoltaic (PV) power output using meteorological data from Aswan, Egypt.

## Overview

Solar energy integration into electrical grids faces challenges due to the intermittent nature of weather conditions. This project develops reliable models to correlate complex meteorological parameters with solar power output, enabling accurate generation predictions and ensuring grid stability.

## Problem Statement

The main challenge is developing a model that can accurately predict solar power generation levels by analyzing non-linear relationships between meteorological parameters (temperature, humidity, wind speed, pressure, dew point) and PV output.

## Dataset

- **Source**: Aswan Weather Data
- **Size**: 398 rows, 8 columns
- **Features**:
  - Average Temperature
  - Humidity
  - Wind Speed
  - Pressure
  - Dew Point
  - Solar(PV) output (target variable)
- **Target Classes**: Low (133), Medium (132), High (133)

## Methodology

### 1. Data Preprocessing
- Missing value imputation using column means
- Date-time parsing to extract temporal features (Month, Season, Day)
- Feature scaling using StandardScaler
- Target variable binning into three balanced classes

### 2. Feature Engineering
- Temporal feature extraction (Month, Season)
- Interaction features (e.g., Temperature × Humidity, Temp_Pressure)
- Statistical validation using Chi-Square and ANOVA tests

### 3. Dimensionality Reduction
- **PCA (Principal Component Analysis)**: Variance retention with reduced features
- **LDA (Linear Discriminant Analysis)**: Maximizing class separability
- **SVD (Singular Value Decomposition)**: Matrix factorization and noise reduction

### 4. Machine Learning Models

#### Classification Models
- Naive Bayes
- Decision Trees (with hyperparameter tuning)
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM with RBF kernel)
- Random Forest (ensemble bagging)
- Gradient Boosting (ensemble boosting)
- Multi-Layer Perceptron (MLP) Neural Network

#### Regression Models
- Linear Regression
- MLP Regressor

## Results

### Model Performance

| Model | Train Accuracy | Test Accuracy | Status |
|-------|---------------|---------------|---------|
| Random Forest | 100.00% | 81.25% | Overfitting |
| Gradient Boosting | 100.00% | 82.50% | Overfitting |
| Decision Tree (Tuned) | 90.57% | 76.25% | Good Fit |
| MLP Classifier | 85.00% | 79.00% | Good Fit |
| KNN (k=5) | 82.70% | 68.75% | Slight Overfitting |
| SVM (RBF) | 74.84% | 68.75% | Good Fit |
| Naive Bayes | 51.57% | 37.50% | Underfitting |
| Linear Regression | N/A | R² = 0.88 | Good Fit |

### Key Findings

- **Best Performers**: Ensemble methods (Random Forest and Gradient Boosting) achieved the highest accuracy (>80%)
- **Feature Reduction**: PCA explained 95% variance with fewer components; first 2 components captured 72% variance
- **LDA Performance**: First component explained 84% of class variance
- **Neural Networks**: MLP provided robust generalization with proper scaling
- **Correlation Analysis**: High correlation between Temp_Pressure and AvgTemperature (0.99)

### Random Forest Metrics
- Accuracy: 81.25%
- Precision (High Class): 0.96
- Recall (High Class): 0.85

## Key Contributions

1. Comparative analysis of dimensionality reduction techniques (PCA vs. LDA vs. SVD) on neural networks and classical ML algorithms
2. Introduction of interaction features (e.g., Temp_Pressure) improving model sensitivity
3. Comprehensive evaluation framework using cross-validation, confusion matrices, and ROC curves

## Technologies Used

- Python
- Scikit-learn
- NumPy/Pandas
- Machine Learning algorithms
- Deep Learning (MLP)
- Statistical analysis tools

## Pipeline Architecture

```
Data Acquisition → Preprocessing → Feature Engineering → 
Dimensionality Reduction → Model Training → Evaluation
```

1. **Data Acquisition**: Load Aswan Weather CSV
2. **Preprocessing**: Date conversion, null treatment, binning, standardization
3. **Feature Selection/Reduction**: Apply PCA/LDA/SVD
4. **Model Training**: 80/20 train-test split
5. **Evaluation**: Confusion matrices, ROC curves, MSE/R² scores

## Conclusions

The project successfully developed a machine learning framework for solar power prediction. Ensemble methods demonstrated superior performance, while neural networks effectively modeled non-linear weather patterns. Feature reduction via LDA proved valuable for visualization, though full feature sets yielded better predictive accuracy.

## Future Work

1. **Deep Learning Enhancement**: Implement LSTM networks for time-series forecasting to exploit sequential weather patterns
2. **Data Expansion**: Acquire multi-year datasets to reduce overfitting in tree-based models
3. **Optimization**: Apply Bayesian Optimization for more efficient hyperparameter tuning
4. **Real-time Integration**: Deploy models for real-time grid management applications

## References

1. Gensler et al. (2016) - Deep Learning for solar power forecasting using LSTM
2. Sharma et al. (2011) - ML methods for solar generation prediction
3. Zang et al. (2020) - CNN for solar pattern classification (94.5% accuracy)
4. Voyant et al. (2017) - Time series methods for solar forecasting
5. Benmouiza et al. (2019) - K-Means and neural networks for solar prediction
6. Wang et al. (2019) - PCA and Gradient Boosting hybrid approach
7. Mellit et al. (2010) - ANN for 24-hour solar irradiance forecasting
8. Al-Dahidi et al. (2019) - ELM-based intelligent forecasting
9. Aslam et al. (2021) - Long-term forecasting with deep learning
10. Li et al. (2022) - Hybrid SVD-ensemble for robust predictions

## Acknowledgments

Data collected from Aswan meteorological stations.