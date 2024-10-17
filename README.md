# Application of ARIMA predictive models and Recurrent Neural Networks (RNN) for pension forecasting

## Overview
This repository contains the code and documentation for my Final Degree Project (TFG), titled "Predictive models for the Spanish pension system in the short and medium term". The project aims to forecast the evolution of the Spanish pension system using two main techniques:

ARIMA (AutoRegressive Integrated Moving Average): A traditional statistical method for time series forecasting.

Recurrent Neural Networks (RNN): A deep learning approach, specifically Long Short-Term Memory (LSTM) networks, to capture complex temporal dependencies.

Both methods are applied to model and predict the pension system's behavior in the short and medium term, providing insights into its future sustainability.

## Project Objectives
- Develop accurate models for predicting the number of pension beneficiaries and the total expenditure.

- Compare traditional time series forecasting (ARIMA) with advanced machine learning techniques (RNN).

- Analyze the implications of different economic and demographic scenarios on the Spanish pension system.

## Data Sources
The data used in this project come from multiple public and reliable sources, including:

- INE (Instituto Nacional de Estadística): Provides demographic data such as population growth, life expectancy, and mortality rates.

- Ministerio de Inclusión, Seguridad Social y Migraciones: Historical data on pension payments and beneficiaries.

- Banco de España: Macroeconomic indicators, such as inflation rates and GDP growth, that impact pension sustainability.

The data is preprocessed and transformed to fit the requirements of both ARIMA and RNN models.

## Repository Structure
📂 ARIMA/               # Contains notebooks and scripts for ARIMA model development

📂 Análisis del dato/    # Data analysis, preprocessing, and feature engineering scripts

📂 RNN/                 # Implementation of Recurrent Neural Networks (LSTM)

.gitignore              # Specifies files to ignore in version control

LICENSE                 # Licensing information

README.md               # This README file

## Key Folders
ARIMA/: This directory includes notebooks and scripts for developing the ARIMA model. It contains time series analysis and model validation steps.

Análisis del dato/: This folder contains scripts for data cleaning, preprocessing, and exploratory data analysis (EDA), which prepare the dataset for both ARIMA and RNN models.

RNN/: The RNN folder focuses on the implementation of the LSTM-based recurrent neural network used for pension forecasting, alongside tuning and evaluation scripts.

## Methodology
1. Data Preprocessing

Cleaning: Handled missing values and outliers.

Feature Engineering: Created new variables such as age groupings and economic indicators.

Normalization: Scaled the data to improve model performance.


2. ARIMA Model

ARIMA is used for short-term, linear time series forecasting. The model is trained after performing stationarity tests and parameter tuning using ACF/PACF plots and grid search for the best (p, d, q) values.

3. Recurrent Neural Networks (RNN - LSTM)

The RNN model, particularly an LSTM network, is designed to handle the non-linear and complex dependencies within time series data. This approach uses sequential input data to predict future trends more accurately for medium-term forecasts.

## Evaluation Metrics
Both models are evaluated using:

- RMSE (Root Mean Square Error)

- MAPE (Mean Absolute Percentage Error)

- R-squared for goodness of fit.

## Results
ARIMA: Performs well in short-term predictions but struggles with longer-term patterns.

RNN/LSTM: Outperforms ARIMA in medium-term forecasting due to its ability to model non-linear dependencies in time series.

## Future Work
- Expand data sources to include more recent trends and additional variables.

- Test more advanced models, such as ensemble learning techniques or hybrid models combining ARIMA and RNN.

- Scenario simulation: Use the models to simulate how different macroeconomic changes could impact pension sustainability.

## Author
Jaime García Sainz de Rozas

Business Analytics & Data Science

University Francisco de Vitoria, Madrid

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
