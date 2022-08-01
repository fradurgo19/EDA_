import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
%matplotlib inline

df = pd.read_excel(r"C:\Users\fadg1\OneDrive\Escritorio\Analytics\PRUEBA TECNICA ANALSTAS DE DATOS\Datos- Modelación/Housing.xlsx", sheet_name='Data')

## ELIMINAR NULOS 
df = df.fillna(0) 

## CAMBIAR OBJECT INT 0 Y 1
gen_ = {"yes" :1, "no" :0}
df["driveway"] = df["driveway"].map(gen_)
df["recroom"] = df["recroom"].map(gen_)
df["fullbase"] = df["fullbase"].map(gen_)
df["gashw"] = df["gashw"].map(gen_)
df["airco"] = df["airco"].map(gen_)
df["prefarea"] = df["prefarea"].map(gen_)
df.head()

## CASAS MAS COMUN SEGUN A DORMITORIO BEDROOMS
df["bedrooms"].value_counts().plot(kind="bar")
plt.title("NUMERO DE HABITACIONES", fontsize =15)
plt.xlabel("Habitaciones", fontsize =15 )
plt.ylabel("Cantidad", fontsize =15 )
plt.xticks(fontsize =15 )
plt.yticks(fontsize =15 )
sns.despine

## CASAS MAS COMUN SEGUN A BAÑOS bathrms
df["bathrms"].value_counts().plot(kind="bar")
plt.title("NUMERO DE BAÑOS", fontsize =15)
plt.xlabel("Baños", fontsize =15 )
plt.ylabel("Cantidad", fontsize =15 )
plt.xticks(fontsize =15 )
plt.yticks(fontsize =15 )
sns.despine

## PRECIO SEGÚN TAMAÑO DEL LOTE
plt.scatter(df.price,df.lotsize)
plt.title("PRECIO VS TAMAÑO DEL LOTE", fontsize =15)
plt.xlabel("Precio", fontsize =15 )
plt.ylabel("Tamño lote", fontsize =15 )
plt.xticks(fontsize =10 )
plt.yticks(fontsize =10 )
sns.despine

## MODELO TECNICA REGRESION LINEAL PERMITE PREDECIR EL COMPORTAMIENTO DE UNA VARIABLE
from sklearn.linear_model import LinearRegression


reg = LinearRegression()

## SE IMPORTAN DATOS PARA REGRESION LINEAL
df1 = pd.read_excel(r"C:\Users\fadg1\OneDrive\Escritorio\Analytics\PRUEBA TECNICA ANALSTAS DE DATOS\Datos- Modelación/Housing.xlsx", sheet_name='Forecast')

## SE DEFINEN VARIABLES PRECIO Y DATOS SIN PRECIO
labels = df["price"]
train1 =  df.drop(["price"], axis=1) ## SE ELIMINA DE DATOS PRECIO


from sklearn.model_selection import train_test_split

## SE CONSTRUYE MODELO DATOS DE ESTRENAMIENTO Y PRUEBA
x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.33, random_state =42)

## SE ENTRENA EL MODELO REGRESION LINEAL
reg.fit(x_train,y_train)

## SE VERIFICA CALDIAD DEL MODELO
reg.score(x_test,y_test)

## EVALUACION PARA LAS PREDICCIONES EN TRAIN DATOS
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

## SE CONSTRUYE MODELO TECNICA DE POTENCIACION DEL GRADIENTE GRADIENTE CONJUNTO DE ARBOLES DE DECISION ANALISIS PREDICTIVO
from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators= 400, max_depth= 5, min_samples_split= 2, 
learning_rate = 0.1, loss = "ls")

## SE ENTRENA EL MODELO GRADIENTE
clf.fit(x_train, y_train)

## SE VERIFICA CALDIAD DEL MODELO
clf.score(x_test,y_test)

## EVALUACION PARA LAS PREDICCIONES EN TRAIN DATOS
training_data_prediction1 = clf.predict(x_train)
print(training_data_prediction1)

## R2 ERROR CAUDRADO
from sklearn import metrics

valor_1 = metrics.r2_score(y_train, training_data_prediction1)

## ERROR MEDIO ABSOLUTO

valor_2 = metrics.mean_absolute_error(y_train, training_data_prediction1)

print("R2 ERROR CAUDRADO : ", valor_1)
print("ERROR MEDIO ABSOLUTO : ", valor_2)

plt.scatter(y_train, training_data_prediction1 )
plt.xlabel("Precios Actuales", fontsize =15 )
plt.ylabel("Prediccion de precios", fontsize =15 )
plt.title("Precios Actuales vs Prediccion de precios", fontsize =15)
plt.show()