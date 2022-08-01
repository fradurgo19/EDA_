import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
%matplotlib inline

df = pd.read_excel(r"C:\Users\fadg1\OneDrive\Escritorio\Analytics\PRUEBA TECNICA ANALSTAS DE DATOS\Datos- Modelación/Housing.xlsx", sheet_name='Data')
df1 = pd.read_excel(r"C:\Users\fadg1\OneDrive\Escritorio\Analytics\PRUEBA TECNICA ANALSTAS DE DATOS\Datos- Modelación/Housing.xlsx", sheet_name='Forecast')

## CAMBIAR OBJECT INT 0 Y 1
gen_ = {"yes" :1, "no" :0}
df["driveway"] = df["driveway"].map(gen_)
df["recroom"] = df["recroom"].map(gen_)
df["fullbase"] = df["fullbase"].map(gen_)
df["gashw"] = df["gashw"].map(gen_)
df["airco"] = df["airco"].map(gen_)
df["prefarea"] = df["prefarea"].map(gen_)

from numpy import int64

## CAMBIAR FLOAT A INT64
df["price"] = df["price"].astype(int64)
df.info()

## CAMBIAR OBJECT INT 0 Y 1
gen_ = {"yes" :1, "no" :0}
df1["driveway"] = df1["driveway"].map(gen_)
df1["recroom"] = df1["recroom"].map(gen_)
df1["fullbase"] = df1["fullbase"].map(gen_)
df1["gashw"] = df1["gashw"].map(gen_)
df1["airco"] = df1["airco"].map(gen_)
df1["prefarea"] = df1["prefarea"].map(gen_)

## REMPLAZAR ESPACIOS VACIOS CON CERO SIN MODIFICAR EL DF
df1.fillna(0, inplace=True)

df1.info()

## SE DEFINE precio
precio = df["price"]
features = df1

## CAMBIAR FLOAT A INT64
## features["recroom"] = features["recroom"].astype(int)

## CONCOES CANTIDAD DE DATOS Y VARIABLES
print("dataframe df tiene {} puntos de datos con {} varibles.".format(*df.shape))

print("dataframe df1 tiene {} puntos de datos con {} varibles.".format(*df1.shape))

## CALCULOS ESATDISTICOS

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

## EXPLORIN DATA ANALITICS EDA
## GRAFICOS DE DISPERSION E HISTOGRAMAS

sns.pairplot(df, size=1)
plt.tight_layout()

## SE OBSERVA UNA RELACION LINEAL ENTRE EL PRECIO (PRICE) Y EL TAMAÑO DE LA VIVIENDA (LOTSIZE)

## CORRELACION DE LAS VARIABLES
correlacion = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlacion, cbar=True, square=True, fmt=".1f", annot= True, annot_kws={"size":8}, cmap="Blues")

## SE EVIDENCIA MAYOR CORRELACION ENTRE LOTSIZE(TAMAÑO) Y PRICE 0.90
## SE EVIDENCIA MAYOR CORRELACION ENTRE STORIES(PLANTAS) Y PRICE 0.77

## MATRIZ DE CORRELACION

from numpy import column_stack


cm = np.corrcoef(df.values.T)
sns.set(font_scale=1)
hm =sns.heatmap(cm,cbar=True,annot=True,square=True,fmt=".2f",annot_kws={"size": 7},)

## SE EVIDENCIA MAYOR CORRELACION ENTRE LOTSIZE(TAMAÑO) Y PRICE 0.90
## SE EVIDENCIA MAYOR CORRELACION ENTRE STORIES(PLANTAS) Y PRICE 0.77

## SE AJUSTA DIMENSION DE DATOS =275
preciofilas = precio[:275]

from sklearn.model_selection import train_test_split

## SE CONSTRUYE MODELO DATOS DE ESTRENAMIENTO Y PRUEBA
x_train , x_test , y_train , y_test = train_test_split(features , preciofilas , test_size = 0.33, random_state =42)

print("Entrenamiento y prueba se completo")

## MODELO TECNICA REGRESION LINEAL PERMITE PREDECIR EL COMPORTAMIENTO DE UNA VARIABLE
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

## EVALUACION
training_data_prediction = reg.predict(x_train)
print(training_data_prediction)

## R2 ERROR CAUDRADO
from sklearn import metrics

valor_1 = metrics.r2_score(y_train, training_data_prediction)

## ERROR MEDIO ABSOLUTO

valor_2 = metrics.mean_absolute_error(y_train, training_data_prediction)

print("R2 ERROR CAUDRADO : ", valor_1)
print("ERROR MEDIO ABSOLUTO : ", valor_2)

plt.scatter(y_train, training_data_prediction )
plt.xlabel("Precios Actuales", fontsize =15 )
plt.ylabel("Prediccion de precios", fontsize =15 )
plt.title("Precios Actuales vs Prediccion de precios", fontsize =15)
plt.show()

X = df1 ## SE DEFINE X VARIABLES DE DF1
Y = df["price"] ## SE DEFINE SOLO PRECIOS DE DF
Y = Y[:275] ## SE CAMBIA LONGITUD A 275

print(Y)

## MODELO CON XGBRegressor
## SE CONSTRUYE MODELO TECNICA DE POTENCIACION DEL GRADIENTE GRADIENTE CONJUNTO DE ARBOLES DE DECISION ANALISIS PREDICTIVO

from sklearn.model_selection import train_test_split

## SE CONSTRUYE MODELO DATOS DE ESTRENAMIENTO Y PRUEBA
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2, random_state =2)

print("Entrenamiento y prueba se completo")

print(X.shape, X_train.shape, X_test.shape)

## MODELO CON XGBOOST REGRESSOR

from xgboost import XGBRegressor

modelo = XGBRegressor()

## ENTRENAR EL MODELO CON X_TRAIN
modelo.fit(X_train, Y_train)

## EVALUACION PARA LAS PREDICCIONES EN TRAIN DATOS
training_data_prediction1 = modelo.predict(X_train)
print(training_data_prediction1)

## R2 ERROR CAUDRADO
from sklearn import metrics

valor_1 = metrics.r2_score(Y_train, training_data_prediction1)

## ERROR MEDIO ABSOLUTO

valor_2 = metrics.mean_absolute_error(Y_train, training_data_prediction1)

print("R2 ERROR CAUDRADO : ", valor_1)
print("ERROR MEDIO ABSOLUTO : ", valor_2)

fig = plt.figure(figsize=(20,20))
plt.scatter(Y_train, training_data_prediction1 )
plt.xlabel("Precios Actuales", fontsize =20 )
plt.ylabel("Prediccion de precios", fontsize =20 )
plt.title("Precios Actuales vs Prediccion de precios", fontsize =20)
plt.show()

## EXPORTAR DATOS PREDICCION
from pandas import DataFrame


prediccion = DataFrame(training_data_prediction1)
prediccion
prediccion.to_excel(r"C:\Users\fadg1\OneDrive\Escritorio\Analytics\PRUEBA TECNICA ANALSTAS DE DATOS\Datos- Modelación/Prediccion.xlsx")

## EVALUACION PARA LAS PREDICCIONES EN TEST DATOS
test_data_prediction1 = modelo.predict(X_test)
print(test_data_prediction1)

## R2 ERROR CAUDRADO
from sklearn import metrics

valor_1 = metrics.r2_score(Y_test, test_data_prediction1)

## ERROR MEDIO ABSOLUTO

valor_2 = metrics.mean_absolute_error(Y_test, test_data_prediction1)

print("R2 ERROR CAUDRADO : ", valor_1)
print("ERROR MEDIO ABSOLUTO : ", valor_2)