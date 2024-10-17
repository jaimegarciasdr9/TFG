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
