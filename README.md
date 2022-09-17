# EDA_Calculo_estadistico_Graficos_Correlacion_Modelo_train_test_LinearRegression_R2_score_Mean_absolute_error_XGBRegressor_RandomForestClassifier


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
10. Regresion Lineal 
11. Evaluaión
12. Gráfica 
13. XGBRegressor
14. Random Forest

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
    
## 6. CALCULOS ESATDISTICOS 🔧

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
  
## 7. EDA 📖
## GRAFICOS DE DISPERSION E HISTOGRAMAS

    sns.pairplot(df, size=1)
    plt.tight_layout()

## SE OBSERVA UNA RELACION LINEAL ENTRE EL PRECIO (PRICE) Y EL TAMAÑO DE LA VIVIENDA (LOTSIZE)  

## 8. CORRELACION DE LAS VARIABLES 📖
SE EVIDENCIA MAYOR CORRELACION ENTRE LOTSIZE(TAMAÑO) Y PRICE 0.90
SE EVIDENCIA MAYOR CORRELACION ENTRE STORIES(PLANTAS) Y PRICE 0.77

    correlacion = df.corr()
    plt.figure(figsize=(10,10))
    sns.heatmap(correlacion, cbar=True, square=True, fmt=".1f", annot= True, annot_kws={"size":8}, cmap="Blues")

## 9. SE CONSTRUYE MODELO DATOS DE ESTRENAMIENTO Y PRUEBA 🔧

    from sklearn.model_selection import train_test_split
    x_train , x_test , y_train , y_test = train_test_split(features , preciofilas , test_size = 0.33, random_state =42)
    print("Entrenamiento y prueba se completo")
  
## 10. MODELO TECNICA REGRESION LINEAL PERMITE PREDECIR EL COMPORTAMIENTO DE UNA VARIABLE 📦

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(x_train,y_train)
  
## 11. EVALUACION 

    training_data_prediction = reg.predict(x_train)
    print(training_data_prediction)

## R2 ERROR CAUDRADO 

    from sklearn import metrics
    valor_1 = metrics.r2_score(y_train, training_data_prediction)

## ERROR MEDIO ABSOLUTO 

    valor_2 = metrics.mean_absolute_error(y_train, training_data_prediction)

    print("R2 ERROR CAUDRADO : ", valor_1)
    print("ERROR MEDIO ABSOLUTO : ", valor_2)
  
## 12. VALIDACION GRAFICA ⚙️

    plt.scatter(y_train, training_data_prediction )
    plt.xlabel("Precios Actuales", fontsize =15 )
    plt.ylabel("Prediccion de precios", fontsize =15 )
    plt.title("Precios Actuales vs Prediccion de precios", fontsize =15)
    plt.show()
  
## 13. MODELO CON XGBRegressor ⚙️
SE CONSTRUYE MODELO TECNICA DE POTENCIACION DEL GRADIENTE GRADIENTE CONJUNTO DE ARBOLES DE DECISION ANALISIS PREDICTIVO 

## MODELO CON XGBOOST REGRESSOR

    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2, random_state =2)
    from xgboost import XGBRegressor
    modelo = XGBRegressor()
    modelo.fit(X_train, Y_train)
  
## 14.RANDOM FOREST ⚙️

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from matplotlib import rcParams
    X = df1 ## SE DEFINE X VARIABLES DE DF1
    Y = df["price"] ## SE DEFINE SOLO PRECIOS DE DF
    Y = Y[:275] ## SE CAMBIA LONGITUD A 275
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)
    from sklearn.ensemble import RandomForestClassifier

      # Crear el modelo con 100 arboles
      model = RandomForestClassifier(n_estimators=100, bootstrap = True,verbose=2, max_features = 'sqrt')
      # entrenar!
      model.fit(X_train, Y_train)
      pred_y = model.predict(X_train)
      ## R2 ERROR CAUDRADO
    from sklearn import metrics

    valor_1 = metrics.r2_score(Y_train, pred_y)

  ## ERROR MEDIO ABSOLUTO

    valor_2 = metrics.mean_absolute_error(Y_train, pred_y)

    print("R2 ERROR CAUDRADO : ", valor_1)
    print("ERROR MEDIO ABSOLUTO : ", valor_2)

## Autor ✒️
    
⭐️ [fradurgo19](https://github.com/fradurgo19)

