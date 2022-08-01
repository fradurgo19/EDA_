## IMPORTAR LLIBRERIAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px  

## IMPORTAR DATOS
df = pd.read_excel(r"C:\Users\fadg1\OneDrive\Escritorio\Analytics\PRUEBA TECNICA ANALSTAS DE DATOS\Datos-Tratamiento de datos y estadística descriptiva/Datos-Empleados Prueba Analista.xlsx")
df1=pd.read_excel(r"C:\Users\fadg1\OneDrive\Escritorio\Analytics\PRUEBA TECNICA ANALSTAS DE DATOS\Datos-Tratamiento de datos y estadística descriptiva/Datos-Empleados Prueba Analista.xlsx", sheet_name='Tabla Empleados') 
df2=pd.read_excel(r"C:\Users\fadg1\OneDrive\Escritorio\Analytics\PRUEBA TECNICA ANALSTAS DE DATOS\Datos-Tratamiento de datos y estadística descriptiva/Datos-Empleados Prueba Analista.xlsx", sheet_name='Tabla Evaluacion')
df3=pd.read_excel(r"C:\Users\fadg1\OneDrive\Escritorio\Analytics\PRUEBA TECNICA ANALSTAS DE DATOS\Datos-Tratamiento de datos y estadística descriptiva/Datos-Empleados Prueba Analista.xlsx", sheet_name='Tabla Sueldo')

## UNIR LOS DF SEGÚN CLAVE ID EMPLEADO METODO JOIN DF1 Y DF2
join = df1.join(df2.set_index('ID Empleado'), on='ID Empleado')

## UNIR LOS DF SEGÚN CLAVE ID EMPLEADO METODO JOIN DF2 Y DF3
join1 = join.join(df3.set_index('ID Empleado'), on='ID Empleado')
join1

## UNIR LOS DF SEGÚN CLAVE ID EMPLEADO METODO PD.MARGE DF1 Y DF2
output1 = pd.merge(df1, df2, on = ['ID Empleado'], how ='inner') 

## UNIR LOS DF SEGÚN CLAVE ID EMPLEADO METODO PD.MARGE output1 Y DF3
output2 = pd.merge(output1, df3, on = ['ID Empleado'], how ='inner') 

##ELIMINAR ESPACIOS EN BLANCO INICIALES O POSTERIORES EN DF
output2.columns = output2.columns.str.strip() 

from numpy import int64

## SE CONVIERTE EN STR
output2["Evaluación"] = output2["Evaluación"].apply(str).str.replace(',', '.') 
output2["Sueldo"] = output2["Sueldo"].apply(str).str.replace(',', '.') 
## CAMBIAR OBJECT A FLOAT
output2["Evaluación"] = output2["Evaluación"].astype(float, errors = "raise")
output2["Sueldo"] = output2["Sueldo"].astype(float, errors = "raise")  

## CAMBIAR FLOAT A INT64
output2["Evaluación"] = output2["Evaluación"].astype(int64)
output2["Sueldo"] = output2["Sueldo"].astype(int64)

output2.info()

## CONOCER EL SUELDO MAS ALTO
output2["Sueldo"][output2["Sueldo"].idxmax()]
output2.describe()

print(output2.dtypes)

print('Cantidad de Filas y columnas:',output2.shape)
print('Nombre columnas:',output2.columns)

output2.head()

## ELIMINAR NULOS 
output2 = output2.fillna(0) 

##GRAFICO DE BARRA ENTRE LA GENERO Y EL SUELDO
from matplotlib.pyplot import legend


x = output2["Género"].values
y = output2["Sueldo"].values
fig = plt.figure(figsize=(10,10))
plt.xlabel("Género", fontsize =15 )
plt.ylabel("Sueldo", fontsize =15 )
plt.xticks(fontsize =15 )
plt.yticks(fontsize =15 )
plt.plot(x, y, "o", color="Purple")

##GRAFICO DE DISPERSION ENTRE LA EVALUCION Y EL SUELDO
x = output2["Evaluación"]
y = output2["Sueldo"]
fig = plt.figure(figsize=(10,10))
plt.scatter(x,y)
plt.xlabel("Evaluación", fontsize =15 )
plt.ylabel("Sueldo", fontsize =15 )
plt.xticks(fontsize =15 )
plt.yticks(fontsize =15 )
plt.show()

## HISTOGRAMA SUELDO
hist_plotsuel= output2["Sueldo"].hist(bins=10)
plt.xlabel("Sueldo", fontsize =15 )
plt.ylabel("Cantidad", fontsize =15 )
plt.title("Histograma Sueldo", fontsize =15)

## HISTOGRAMA EVALUACION
hist_ploteva= output2["Evaluación"].hist(bins=2)
plt.xlabel("Evaluación", fontsize =15 )
plt.ylabel("Cantidad", fontsize =15 )
plt.title("Histograma Evaluación", fontsize =15)

output3 = output2.loc[:, ["Género","Evaluación","Sueldo"]] ##Seleccionar solo unas columnas
output3 
generoysueldo_profit_bar = px.bar(output3,x="Evaluación",y="Género",color="Sueldo",color_continuous_scale=["red", "yellow", "green"],
    title="<b>Genero & sueldo</b>",
)
generoysueldo_profit_bar.show()

genero_profit_scatter = px.scatter( output3, x="Sueldo", y="Género", color="Evaluación",title="<b>Genero/Sueldo</b>")
genero_profit_scatter.show()

output2.Estado.str.split(",",expand=True) ##separar

Estado1 = output2.Estado.str.split(",",expand=True)[0] ##Separa datos de una columna y agregarlos al dataframe
Pais = output2.Estado.str.split(",",expand=True)[1]
output2["Estado1"] = Estado1
output2["Pais"] = Pais

output2.head()

pd.crosstab(output2["Estado"], columns= "count").sort_values(by="count", ascending=False) ##verificar cuantas veces se repite Estado

pd.crosstab(output2["Posición"], columns= "count").sort_values(by="count", ascending=False)

(output2["Sueldo"] / 1e+04).describe()

output2[["Evaluación","Sueldo"]].corr()

## CAMBIAR OBJECT INT 0 Y 1
gen_ = {"Male" :1, "Female" :0}
output3["Género"] = output3["Género"].map(gen_)
output3

output3[["Género","Evaluación","Sueldo"]].corr()

output3.Género.value_counts() ##Cantidad Hombre Male =1 mujeres Female = 0

output3.Sueldo.value_counts()

output2genero = output2.groupby(["Género"])["Sueldo"].mean()
output2genero

output2.groupby(["Género"])["Sueldo"].min()

output2.groupby(["Género"])["Evaluación"].mean()

output2.groupby(["Sueldo"]).filter( lambda x: len(x) > 200).Sueldo

DFgrafica = output2.groupby(["Género","Evaluación"])["Sueldo"].sum()

output2.groupby(["Género","Sueldo"])["Evaluación"].mean()

output2.Sueldo >40000

(output2.Sueldo >=1000000).sum() ##Sueldo mayor o igual a 70000

output2[output2.Sueldo == 1000000]

output2.index[output2.Sueldo >=1000000] ##en el dataframe la ubicación mayores o iguales a 1000000

output2[ (output2.Sueldo > 1000000) & (output2.Género   == "Male"  ) ] ## Cueldos mayores a 1000000 y sean hombres

output2.Sueldo.max() ## el mayor sueldo

output2[output2.Sueldo == output2.Sueldo.max()] ## seleccionar sueldos mas altos