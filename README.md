# Predicting Appliances Energy Consumption

## Project Overview
This project focuses on predicting appliances' energy consumption in a low-energy building based on environmental, temporal, and weather variables. The goal is to provide actionable insights into energy usage patterns and support energy optimization strategies. Machine learning models were developed to predict and classify energy consumption levels (high, average, or low) using a dataset collected from ZigBee sensors and weather stations.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Future Work](#future-work)
8. [References](#references)
9. [Contributors](#contributors)

## Dataset
The dataset consists of 19,735 hourly records collected from Chievres Airport, Belgium. It includes key input variables such as indoor temperature, humidity, outdoor temperature, wind speed, and visibility. The target variable is the energy consumption of appliances, measured in watt-hours (Wh).

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/your-dataset-link).

## Installation
To run the code in this repository, you need to have Python 3.x installed along with the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost dmba

```

## Clone the repository
```bash
git clone https://github.com/your-username/predicting-energy-consumption.git
cd predicting-energy-consumption

```

## Usage
1. Data Preprocessing: Run the data_preprocessing.py script to clean and preprocess the data.
2.Model Training: Use the model_training.py script to train the Random Forest Regressor and other models.
3.Evaluation: Evaluate the models using the evaluation.py script, which generates metrics like R², RMSE, and MAE.
```bash
python data_preprocessing.py
python model_training.py
python evaluation.py
```

## Methodology
1.Data Collection and Processing: The dataset was imported and inspected for missing values. Statistical summaries were generated to understand data distributions.
2.Data Exploration and Visualization: Correlation heatmaps were created to assess relationships between variables. Redundant variables were removed to reduce multicollinearity.
3.Feature Engineering: Key predictors such as outdoor temperature and humidity were identified. Random Forest was used to assess feature importance.
4.Model Selection: Various regression models were tested, including Linear Regression, Random Forest Regressor, ARIMA, Lasso, Ridge, XGBoost, and MLPRegressor. Random Forest Regressor consistently outperformed other models.
5.Evaluation: Cross-validation and metrics such as R², RMSE, and MAE were used for evaluation.

## Results
The Random Forest Regressor demonstrated strong performance, with high R² scores and low RMSE and MAE scores. The correlation between the actual and predicted values of the target variable was found to be 0.68, indicating a moderate level of prediction accuracy.

Actual vs Fitted Values
