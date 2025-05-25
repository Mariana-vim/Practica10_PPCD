import pandas as pd

# Ruta del archivo CSV
file_path = './emisiones-2018.csv'

# Detectar delimitador automáticamente
with open(file_path, 'r', encoding='utf-8') as f:
    primera_linea = f.readline()
    delimitador = ';' if primera_linea.count(';') > primera_linea.count(',') else ','

# Leer el archivo CSV con el delimitador adecuado
df = pd.read_csv(file_path, delimiter=delimitador)

# Crear un DataFrame con nombre de columna, tipo de dato y algunos valores de ejemplo
estructura = pd.DataFrame({
    'Columna': df.columns,
    'Tipo de dato': df.dtypes.values,
    'Ejemplo 1': df.iloc[0].values,
    'Ejemplo 2': df.iloc[1].values if len(df) > 1 else [''] * len(df.columns),
    'Ejemplo 3': df.iloc[2].values if len(df) > 2 else [''] * len(df.columns),
})

# Mostrar la estructura
print(estructura)
