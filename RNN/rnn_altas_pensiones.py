from rnn_essentials import (load_data,
                             check_missing_values,
                             create_output_dir,
                             rnn_model
                             )

if __name__ == "__main__":
    file_path = "C:/Users/jaime/Desktop/TFG/Presentación Análisis de negocio/data/pensiones_dataset.xlsx"
    df = load_data(file_path)
    print(df.head())
    output_dir = "C:/Users/jaime/Desktop/TFG/Presentación Análisis del dato/graficos2/altas_pensiones2/prueba"
    column_name = "num_altas_pensiones"
    # ------------------------------
    n_steps = 3
    n_units = 10
    n_features = 10
    # ------------------------------
    check_missing_values(df)
    create_output_dir(output_dir)
    rnn_model(df, column_name, output_dir)


