import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel(r"C:\Users\fadg1\OneDrive\Escritorio\Analytics\PRUEBA TECNICA ANALSTAS DE DATOS\Datos-Tratamiento de datos y estadística descriptiva/Datos-Empleados Prueba Analista.xlsx")
df1=pd.read_excel(r"C:\Users\fadg1\OneDrive\Escritorio\Analytics\PRUEBA TECNICA ANALSTAS DE DATOS\Datos-Tratamiento de datos y estadística descriptiva/Datos-Empleados Prueba Analista.xlsx", sheet_name='Tabla Empleados') 
df2=pd.read_excel(r"C:\Users\fadg1\OneDrive\Escritorio\Analytics\PRUEBA TECNICA ANALSTAS DE DATOS\Datos-Tratamiento de datos y estadística descriptiva/Datos-Empleados Prueba Analista.xlsx", sheet_name='Tabla Evaluacion')
df3=pd.read_excel(r"C:\Users\fadg1\OneDrive\Escritorio\Analytics\PRUEBA TECNICA ANALSTAS DE DATOS\Datos-Tratamiento de datos y estadística descriptiva/Datos-Empleados Prueba Analista.xlsx", sheet_name='Tabla Sueldo')

join = df1.join(df2.set_index('ID Empleado'), on='ID Empleado')

join1 = join.join(df3.set_index('ID Empleado'), on='ID Empleado')

join1

union = df1.set_index('ID Empleado').join(df2.set_index('ID Empleado'))

union

output1 = pd.merge(df1, df2, on = ['ID Empleado'], how ='inner') 

output2 = pd.merge(output1, df3, on = ['ID Empleado'], how ='inner') 

output2.info()

output2.describe()

print(output2.dtypes)

print('Cantidad de Filas y columnas:',output2.shape)
print('Nombre columnas:',output2.columns)

output2.head()

output2 = output2.fillna(0)