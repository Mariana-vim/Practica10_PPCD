# Practica 10 - Visualizaciones de datos Titanic y Emisiones
# despues de realizar (tarde) la practica 9 de pandas con graficas personalizadas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
colors_palette = ['#9370DB', '#8A2BE2', '#9932CC', '#BA55D3', '#DA70D6', '#DDA0DD']

print("Practica 9: Analisis de Datos")
print("\nNo.1. Analisis del dataset titanic")

df_titanic = pd.read_csv('titanic.csv')

# mostrar informacion basica del dataframe
print(f"dimensiones del dataframe: {df_titanic.shape}")
print(f"numero total de datos: {df_titanic.size}")
print(f"nombres de columnas: {list(df_titanic.columns)}")
print(f"nombres de filas (primeras 5): {list(df_titanic.index[:5])}")
print("\ntipos de datos de las columnas:")
print(df_titanic.dtypes)

print("\nprimeras 10 filas:")
print(df_titanic.head(10))

print("\nultimas 10 filas:")
print(df_titanic.tail(10))

# mostrar datos de un pasajero cualquiera (pasajero 148)
print(f"\ndatos del pasajero con ID 148:")
if 148 in df_titanic.index:
    print(df_titanic.loc[148])
else:
    print("pasajero 148 no encontrado")

# mostrar filas pares
print(f"\nfilas pares del dataframe (primeras 10):")
filas_pares = df_titanic.iloc[::2]
print(filas_pares.head(10))

# nombres de personas en primera clase ordenados alfabeticamente
primera_clase = df_titanic[df_titanic['Pclass'] == 1]['Name'].sort_values()
print(f"\nnombres en primera clase (primeros 10):")
print(primera_clase.head(10).values)

# porcentaje de supervivencia
supervivencia = df_titanic['Survived'].value_counts(normalize=True) * 100
print(f"\nporcentaje de supervivencia:")
print(f"murieron: {supervivencia[0]:.2f}%")
print(f"sobrevivieron: {supervivencia[1]:.2f}%")

# porcentaje de supervivencia por clase
supervivencia_clase = df_titanic.groupby('Pclass')['Survived'].agg(['count', 'sum'])
supervivencia_clase['porcentaje'] = (supervivencia_clase['sum'] / supervivencia_clase['count']) * 100
print(f"\nporcentaje de supervivencia por clase:")
print(supervivencia_clase['porcentaje'])

# eliminar pasajeros con edad desconocida
df_titanic_sin_nan = df_titanic.dropna(subset=['Age'])
print(f"\nfilas despues de eliminar edades desconocidas: {df_titanic_sin_nan.shape[0]}")

# edad media de mujeres por clase
edad_mujeres_clase = df_titanic_sin_nan[df_titanic_sin_nan['Sex'] == 'female'].groupby('Pclass')['Age'].mean()
print(f"\nedad media de mujeres por clase:")
print(edad_mujeres_clase)

# añadir columna de menor de edad
df_titanic_sin_nan['menor_edad'] = df_titanic_sin_nan['Age'] < 18

# porcentaje de menores y mayores que sobrevivieron por clase
menores_supervivencia = df_titanic_sin_nan.groupby(['Pclass', 'menor_edad'])['Survived'].agg(['count', 'sum'])
menores_supervivencia['porcentaje'] = (menores_supervivencia['sum'] / menores_supervivencia['count']) * 100
print(f"\nporcentaje de supervivencia menores/mayores por clase:")
print(menores_supervivencia['porcentaje'])

print("\n\nNo.2. analisis de emisiones contaminantes madrid")

años = [2016, 2017, 2018, 2019]
dfs_emisiones = []

for año in años:
    try:
        df = pd.read_csv(f'emisiones-{año}.csv', sep=';') # indicando separacion por ';'
        print(f"columnas en emisiones-{año}: {list(df.columns[:10])}...")  # mostrar primeras 10
        dfs_emisiones.append(df)
    except FileNotFoundError:
        print(f"archivo emisiones-{año}.csv")
        continue
    except Exception as e:
        print(f"error cargando emisiones-{año}.csv: {e}")
        continue

if dfs_emisiones:
    df_emisiones = pd.concat(dfs_emisiones, ignore_index=True)
    
    # identificar columnas disponibles
    print(f"todas las columnas disponibles: {list(df_emisiones.columns[:15])}...")  # mostrar primeras 15
    
    # buscar columnas que contengan informacion de dias
    columnas_dias = [col for col in df_emisiones.columns if col.startswith('D') and len(col) == 3 and col[1:].isdigit()]
    columnas_validacion = [col for col in df_emisiones.columns if col.startswith('V') and len(col) == 3 and col[1:].isdigit()]
    print(f"columnas de dias encontradas: {len(columnas_dias)} - {columnas_dias[:5]}")
    print(f"columnas de validacion encontradas: {len(columnas_validacion)} - {columnas_validacion[:5]}")
    
    print(f"años unicos: {sorted(df_emisiones['ANO'].unique())}")
    print(f"meses unicos: {sorted(df_emisiones['MES'].unique())}")
    print(f"estaciones unicas: {len(df_emisiones['ESTACION'].unique())} estaciones")
    print(f"magnitudes unicas: {sorted(df_emisiones['MAGNITUD'].unique())}")
    
    if columnas_dias:
        print(f"reestructurando {len(columnas_dias)} columnas de dias")
        
        # identificar columnas id
        id_vars = ['ANO', 'MES', 'ESTACION', 'MAGNITUD', 'PROVINCIA', 'MUNICIPIO', 'PUNTO_MUESTREO']
        id_vars_disponibles = [col for col in id_vars if col in df_emisiones.columns]
        
        print(f"columnas id para reestructurar: {id_vars_disponibles}")
    
        print("realizando 'melt' de datos: el método melt() en Pandas se utiliza/npara transformar un DataFrame de un formato ancho (wide) a un formato largo (long).")
        df_long = pd.melt(df_emisiones, 
                          id_vars=id_vars_disponibles,
                          value_vars=columnas_dias,
                          var_name='DIA_STR',
                          value_name='EMISION')
        
        # extraer numero de dia (con formato de una letra mayuscula y dos numeros) con expresiones regulares
        df_long['DIA'] = df_long['DIA_STR'].str.extract(r'(\d+)').astype(int)
        
        
        # crear mapeo de validación de forma eficiente
        for dia_col in columnas_dias[:10]:  # limitamos a los primeros 10 dias para acelerar
            validacion_col = dia_col.replace('D', 'V')
            if validacion_col in df_emisiones.columns:
                # crear melt para esta validación específica
                df_val_temp = pd.melt(df_emisiones,
                                     id_vars=id_vars_disponibles,
                                     value_vars=[validacion_col],
                                     var_name='VAL_STR',
                                     value_name='VAL_CODE')
                df_val_temp['DIA_STR'] = dia_col
                
                # merge con df_long
                merge_cols = id_vars_disponibles + ['DIA_STR']
                df_long = df_long.merge(df_val_temp[merge_cols + ['VAL_CODE']], 
                                       on=merge_cols, how='left')
                
                # actualizar validación
                mask = pd.notna(df_long['VAL_CODE'])
                df_long.loc[mask, 'VALIDACION'] = df_long.loc[mask, 'VAL_CODE']
                df_long = df_long.drop('VAL_CODE', axis=1)
        
        # filtrar solo datos válidos
        print("filtrando datos válidos...")
        df_long = df_long[df_long['VALIDACION'] == 'V'].copy()
        df_long = df_long[pd.notna(df_long['EMISION'])].copy()
        df_long = df_long[df_long['EMISION'] >= 0].copy()  # eliminar valores negativos
        
        # crear columna fecha de forma vectorizada
        print("creando fechas...")
        try:
            df_long['FECHA'] = pd.to_datetime(df_long[['ANO', 'MES', 'DIA']], errors='coerce')
            df_long = df_long.dropna(subset=['FECHA'])
        except:
            print("error creando fechas, usando fechas simplificadas...")
            df_long['FECHA'] = df_long.apply(lambda x: f"{x['ANO']}-{x['MES']:02d}-{x['DIA']:02d}", axis=1)
        
        # ordenar datos
        df_long = df_long.sort_values(['ESTACION', 'MAGNITUD', 'FECHA'])
        
        print(f"datos reestructurados: {df_long.shape}")
        if 'FECHA' in df_long.columns and hasattr(df_long['FECHA'].iloc[0], 'year'):
            print(f"rango de fechas: {df_long['FECHA'].min()} a {df_long['FECHA'].max()}")
        print(f"rango de emisiones: {df_long['EMISION'].min():.2f} a {df_long['EMISION'].max():.2f}")
        
    else:
        print("no se encontraron columnas dedias validas")
        df_long = pd.DataFrame()
else:
    print("no se pudieron cargar archivos de emisiones")
    df_long = pd.DataFrame()

print("Visualizaciones de Datos:")

# crear figura con subplots para multiples graficas
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Análisis de Datos Titanic y Emisiones Madrid', fontsize=16, color='#4B0082')

# grafica 1: supervivencia por clase (barras)
supervivencia_data = df_titanic.groupby('Pclass')['Survived'].mean()
axes[0,0].bar(supervivencia_data.index, supervivencia_data.values, 
              color=colors_palette[:3], alpha=0.8)
axes[0,0].set_title('Supervivencia por Clase - Titanic', color='#4B0082')
axes[0,0].set_xlabel('Clase')
axes[0,0].set_ylabel('Tasa de Supervivencia')
axes[0,0].set_ylim(0, 1)
for i, v in enumerate(supervivencia_data.values):
    axes[0,0].text(i+1, v+0.02, f'{v:.2f}', ha='center', va='bottom')

# grafica 2: distribucion de edades por sexo (histograma)
df_titanic_clean = df_titanic.dropna(subset=['Age'])
hombres_edad = df_titanic_clean[df_titanic_clean['Sex'] == 'male']['Age']
mujeres_edad = df_titanic_clean[df_titanic_clean['Sex'] == 'female']['Age']

axes[0,1].hist([hombres_edad, mujeres_edad], bins=20, alpha=0.7, 
               color=[colors_palette[0], colors_palette[2]], label=['Hombres', 'Mujeres'])
axes[0,1].set_title('Distribución de Edades por Sexo', color='#4B0082')
axes[0,1].set_xlabel('Edad')
axes[0,1].set_ylabel('Frecuencia')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# grafica 3: tarifa vs edad scatter
scatter = axes[0,2].scatter(df_titanic_clean['Age'], df_titanic_clean['Fare'], 
                           c=df_titanic_clean['Pclass'], cmap='plasma', alpha=0.6)
axes[0,2].set_title('Relación Edad vs Tarifa por Clase', color='#4B0082')
axes[0,2].set_xlabel('Edad')
axes[0,2].set_ylabel('Tarifa')
plt.colorbar(scatter, ax=axes[0,2], label='Clase')

# preparar datos de emisiones para graficas
if not df_long.empty and 'EMISION' in df_long.columns:
    print("creando gráficas de emisiones...")
    
    # grafica 4: emisiones promedio por mes
    if 'MES' in df_long.columns:
        emisiones_mes = df_long.groupby('MES')['EMISION'].mean()
        axes[1,0].plot(emisiones_mes.index, emisiones_mes.values, 
                       marker='o', color=colors_palette[1], linewidth=2, markersize=6)
        axes[1,0].set_title('Emisiones Promedio por Mes', color='#4B0082')
        axes[1,0].set_xlabel('Mes')
        axes[1,0].set_ylabel('Emisión Promedio')
        axes[1,0].set_xticks(range(1,13))
    else:
        # grafica alternativa: distribucion general de emisiones
        emisiones_filtradas = df_long['EMISION'].dropna()
        if len(emisiones_filtradas) > 0:
            axes[1,0].hist(emisiones_filtradas, bins=30, color=colors_palette[1], alpha=0.7)
            axes[1,0].set_title('Distribución de Emisiones', color='#4B0082')
            axes[1,0].set_xlabel('Nivel Emisión')
            axes[1,0].set_ylabel('Frecuencia')
    axes[1,0].grid(True, alpha=0.3)
    
    # grafica 5: boxplot emisiones por año
    if 'ANO' in df_long.columns and len(df_long['ANO'].unique()) > 1:
        años_disponibles = sorted(df_long['ANO'].unique())
        datos_años = [df_long[df_long['ANO'] == año]['EMISION'].dropna().values for año in años_disponibles]
        # filtrar años con datos
        datos_años_filtrados = [datos for datos in datos_años if len(datos) > 0]
        años_filtrados = [año for i, año in enumerate(años_disponibles) if len(datos_años[i]) > 0]
        
        if datos_años_filtrados:
            bp = axes[1,1].boxplot(datos_años_filtrados, labels=años_filtrados, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors_palette):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[1,1].set_title('Distribución Emisiones por Año', color='#4B0082')
            axes[1,1].set_xlabel('Año')
            axes[1,1].set_ylabel('Nivel de Emisión')
    
    # grafica 6: top contaminantes
    if 'MAGNITUD' in df_long.columns:
        top_emisiones = df_long.groupby('MAGNITUD')['EMISION'].mean().sort_values(ascending=False).head(8)
        if len(top_emisiones) > 0:
            y_pos = np.arange(len(top_emisiones))
            axes[1,2].barh(y_pos, top_emisiones.values, color=colors_palette[:len(top_emisiones)])
            axes[1,2].set_yticks(y_pos)
            axes[1,2].set_yticklabels([f'Mag {int(mag)}' for mag in top_emisiones.index])
            axes[1,2].set_title('Top Contaminantes por Emisión Promedio', color='#4B0082')
            axes[1,2].set_xlabel('Emisión Promedio')
            axes[1,2].grid(True, alpha=0.3, axis='x')

# ajustar layout y mostrar visualizacion de graficos final
plt.tight_layout(pad=3)
plt.show()

print("1. Supervivencia por clase Titanic (barras)")
print("2. Distribución edades por sexo (histograma)")  
print("3. Relación edad vs tarifa por clase (scatter)")
if not df_long.empty:
    print("4. Emisiones promedio por mes (linea)")
    print("5. Distribucion emisiones por año (boxplot)")
    print("6. Top contaminantes por emisión promedio (barras)")
