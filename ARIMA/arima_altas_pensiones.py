from arima_utils_2 import (load_data,
                           visualize_data,
                           test_stationarity,
                           plot_acf_comparison,
                           plot_pacf_comparison,
                           plot_decomposition_comparison,
                           differencing_order_1_seasonal_differencing,
                           arima_model,
                           auto_arima_model,
                           train_arima_model,
                           plot_diagnostics,
                           calculate_metrics,
                           plot_predictions)

if __name__ == "__main__":
    file_path = "C:/Users/jaime/Desktop/TFG/Presentación Análisis de negocio/data/pensiones_dataset.xlsx"
    df = load_data(file_path)
    print(df.head())
    output_dir = "C:/Users/jaime/Desktop/TFG/Presentación Análisis del dato/graficos2/altas_pensiones2/ARIMA"
    column_name = "num_altas_pensiones"
    # ------------------------------
    test_stationarity(df, column_name, output_dir)
    plot_acf_comparison(df, column_name, output_dir)
    plot_pacf_comparison(df, column_name, output_dir)
    plot_decomposition_comparison(df, column_name, output_dir)
    differencing_order_1_seasonal_differencing(df, column_name)
    auto_arima_model(df, column_name)
    arima_model(df, column_name, output_dir)
    visualize_data(df, column_name, output_dir)
    model_fit, datos_train, datos_test, model_predictions = train_arima_model(df, column_name)
    plot_diagnostics(model_fit, output_dir)
    plot_predictions(model_predictions, datos_test, output_dir)
    calculate_metrics(df, model_predictions, column_name)

