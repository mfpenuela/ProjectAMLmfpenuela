# ProjectAMLmfpenuela

Los resultados del test no son iguales a los que aparecen en el informe, porque para sacar las metricas utilice la implemetacion oficial del Challenge de FeTa, pero esta es una aplicacion que corre por consola y solo tiene build para windows y ubuntu. No puede replicar la implementacion que ellos usan en el servidor porque definen predicciones suaves, entonces utilice un dice normal. 

Sin embargo, el codigo tambien guarda las prediccione como volumenes .nii si quieren intentar correlo. Seria necesario que descargaran las carpetas de las prediciones:

/home/mfpenuela/FetaProjectAML/Results/testVolsAttention
/home/mfpenuela/FetaProjectAML/Results/testVolsBase

La carpeta del groundTruth:

/home/mfpenuela/FetaProjectAML/Test/masksVolumes

y instalar la aplicacion: https://github.com/Visceral-Project/EvaluateSegmentation. En la carpeta /buids estan los .zip para descargarla. 

En el codigo - estan las lineas para correr todos los volumenes y obtener la prediccion para el conjunto de Test, pero seri necesario que cambiarn los paths a donde se encuentra el ejecutable de la aplicacion y las carpetas de los volumenes. 

