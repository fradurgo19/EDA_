# EDA_Calculo_esatdistico_Graficos_Correlacion_Modelo_train_test_LinearRegression_R2_score_Mean_absolute_error_XGBRegressor_RandomForestClassifier


## Pasos 🚀

1. Se importan librerias 
2. Se seleccionan los datos
3. Se cambia onject
4. Eliminar nulos
5. Se variables test y train
6. Calculos estadístico 
7. EDA
8. Correlación
9. Modelo entrenamiento y prueba
10. Asimetría 
12. KNeighborsClassifier-accuracy_score.

## 1. Importar librerías 🔧

  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  import mpl_toolkits
  %matplotlib inline
  
## 2. Importar Datos 🔧
  
  df = pd.read_excel(r"C:\Users\fadg1\OneDrive\Escritorio\Analytics\PRUEBA TECNICA ANALSTAS DE DATOS\Datos- Modelación/Housing.xlsx", sheet_name='Data')
  df1 = pd.read_excel(r"C:\Users\fadg1\OneDrive\Escritorio\Analytics\PRUEBA TECNICA ANALSTAS DE DATOS\Datos- Modelación/Housing.xlsx", sheet_name='Forecast')

## 3. CAMBIAR OBJECT INT 0 Y 1 🔧

  gen_ = {"yes" :1, "no" :0}
  df["driveway"] = df["driveway"].map(gen_)
  df["recroom"] = df["recroom"].map(gen_)
  df["fullbase"] = df["fullbase"].map(gen_)
  df["gashw"] = df["gashw"].map(gen_)
  df["airco"] = df["airco"].map(gen_)
  df["prefarea"] = df["prefarea"].map(gen_)
  
## 4. REMPLAZAR ESPACIOS VACIOS CON CERO SIN MODIFICAR EL DF 🔧

  df1.fillna(0, inplace=True)

## 5. SE DEFINE precio 🔧

  precio = df["price"]
  features = df1
## 6. CALCULOS ESATDISTICOS

## PRECIO MINIMO
  precio_minimo = np.amin(precio)
## PRECIO MAXIMO
  precio_maximo = np.amax(precio)
## PROMEDIO DE PRECIO
  precio_promedio = np.mean(precio)
## MEDIANA DE PRECIO
  precio_mediana = np.median(precio)
## DESVIACION ESTANDAR DE PRECIO
  precio_desviaciostda = np.std(precio)

  print("Precio minimo: ${}".format(precio_minimo))
  print("Precio maximo: ${}".format(precio_maximo))
  print("Precio promedio: ${}".format(precio_promedio))
  print("Precio mediana: ${}".format(precio_mediana))
  print("Precio desviacion std: ${}".format(precio_desviaciostda))
  
## 7. EXPLORIN DATA ANALITICS EDA
## GRAFICOS DE DISPERSION E HISTOGRAMAS

  sns.pairplot(df, size=1)
  plt.tight_layout()

## SE OBSERVA UNA RELACION LINEAL ENTRE EL PRECIO (PRICE) Y EL TAMAÑO DE LA VIVIENDA (LOTSIZE)  

## 8. CORRELACION DE LAS VARIABLES
SE EVIDENCIA MAYOR CORRELACION ENTRE LOTSIZE(TAMAÑO) Y PRICE 0.90
SE EVIDENCIA MAYOR CORRELACION ENTRE STORIES(PLANTAS) Y PRICE 0.77

  correlacion = df.corr()
  plt.figure(figsize=(10,10))
  sns.heatmap(correlacion, cbar=True, square=True, fmt=".1f", annot= True, annot_kws={"size":8}, cmap="Blues")
  
## 9. SE CONSTRUYE MODELO DATOS DE ESTRENAMIENTO Y PRUEBA  

  from sklearn.model_selection import train_test_split
  x_train , x_test , y_train , y_test = train_test_split(features , preciofilas , test_size = 0.33, random_state =42)
  print("Entrenamiento y prueba se completo")

## Autor ✒️
    
⭐️ [fradurgo19](https://github.com/fradurgo19)

