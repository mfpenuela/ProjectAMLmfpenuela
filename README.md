# ProjectAMLmfpenuela

1. Descargar la carpeta FetaProjectAML y copiar las carpetas con las imagenes de Test y los checkpoints:
```bash
/home/mfpenuela/FetaProjectAML/checkpoint
/home/mfpenuela/FetaProjectAML/Test
```

o pueden pueden copiar toda la carpeta de una vez:

/home/mfpenuela/FetaProjectAML
```
2. Entrar a la carpeta 

cd FetaProjectAML

3. Correr el requeriments.txt

pip install -r requirements.txt

4. Correr el archivo main.py donde el argumento --mode puede ser 'test' o 'demo'

ej: python3 test.py --mode 'test'

El demo tiene el parametro --img que puede ser el nombre del paciente y tomar los valores: 'sub-016', 'sub-023', 'sub-024','sub-027','sub-028','sub-032','sub-035','sub-036','sub-059','sub-064'

ej: python3 test.py --mode 'demo' --img 'sub-023'


Los resultados del test no son iguales a los que aparecen en el informe, porque para sacar las metricas utilice la implemetacion oficial del Challenge de FeTa, pero esta es una aplicacion que corre por consola y solo tiene build para windows y ubuntu. No puede replicar la implementacion que ellos usan en el servidor porque definen predicciones suaves, entonces utilice un dice normal. 

Sin embargo, el codigo tambien guarda las prediccione como volumenes .nii si quieren intentar correlo. Seria necesario que descargaran las carpetas de las prediciones:

/home/mfpenuela/FetaProjectAML/Results/testVolsAttention
/home/mfpenuela/FetaProjectAML/Results/testVolsBase

La carpeta del groundTruth:

/home/mfpenuela/FetaProjectAML/Test/masksVolumes

y instalar la aplicacion: https://github.com/Visceral-Project/EvaluateSegmentation. En la carpeta /buids estan los .zip para descargarla. 

En el codigo - estan las lineas para correr todos los volumenes y obtener la prediccion para el conjunto de Test, pero seri necesario que cambiarn los paths a donde se encuentra el ejecutable de la aplicacion y las carpetas de los volumenes. 

