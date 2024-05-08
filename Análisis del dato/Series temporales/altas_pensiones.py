from statistics_utils import (print_descriptive_statistics,
                              main,
                              load_data,
                              plot_time_series,
                              plot_scatter_plot,
                              plot_boxplot,
                              plot_rolling_mean,
                              plot_rolling_variance,
                              plot_decomposition,
                              test_stationarity)

if __name__ == "__main__":
    file_path = "C:/Users/jaime/Desktop/TFG/Presentación Análisis del dato/data/altas&bajas_pensiones_sin2023.xlsx"
    df = load_data(file_path)
    print(df.head())
    # PARA DEFINIR ------------------
    column_name = "Altas pensiones"
    output_dir = "C:/Users/jaime/Desktop/TFG/Presentación Análisis del dato/graficos/altas_pensiones"
    label_height = 0.99
    column_name_2 = "Bajas pensiones"
    # -------------------------------
    main(df,column_name, output_dir, label_height, column_name_2)
    print_descriptive_statistics(df, column_name)
    plot_time_series(df, column_name, output_dir, label_height)
    plot_scatter_plot(df, column_name, output_dir, label_height)
    plot_boxplot(df, output_dir, column_name)
    plot_rolling_mean(df, column_name, output_dir)
    plot_rolling_variance(df, column_name, output_dir)
    plot_decomposition(df, column_name, output_dir)
    test_stationarity(df, column_name)






