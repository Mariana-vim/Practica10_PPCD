import pandas as pd

# Ruta del archivo CSV
file_path = './titanic.csv'

# Detectar delimitador automáticamente
with open(file_path, 'r', encoding='utf-8') as f:
    primera_linea = f.readline()
    delimitador = ';' if primera_linea.count(';') > primera_linea.count(',') else ','

# Leer el archivo CSV con el delimitador adecuado
df = pd.read_csv(file_path, delimiter=delimitador)

# Mostrar todos los valores por columna en una sola línea
for col in df.columns:
    tipo = df[col].dtype
    valores = df[col].astype(str).tolist()
    print(f'Columna: {col}')
    print(f'Tipo de dato: {tipo}')
    print('Valores: ' + ', '.join(valores))
    print('-' * 80)  # Separador visual
