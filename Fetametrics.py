##
import os
import matplotlib.pyplot as plt
import numpy as np
import statistics


#---------------------------------
#Cambiar paths

labels = os.listdir('--path predicciones--')
segs = os.listdir('--path mascaras--')

for i in range(len(labels)):
    command = r"--path ejexutable-- --path predicciones--/"+labels[i]+" --path mascaras/"+segs[i]+"  -use DICE -xml resultados"+str(i)+".xhtml"  # The command needs to be a string
    os.system(command)
    print(i)

#Ejemplo
##
# labels = os.listdir('voltestsizekl5')
# segs = os.listdir('volsgt')
#
# for i in range(len(labels)):
#     command = r"C:\Users\Usuario\PycharmProjects\pythonProject1\EvaluateSegmentation\builds\Windows\EvaluateSegmentation.exe voltestsizekl5/"+labels[i]+" volsgt/"+segs[i]+"  -use DICE,VOLSMTY,HDRFDST@0.95@ -xml resultados"+str(i)+".xhtml"  # The command needs to be a string
#     os.system(command)
#     print(i)

#----------------------------------

from xml.dom import minidom
dice=[]
for i in range(10):
    doc = minidom.parse("resultados"+str(i)+".xhtml")
    a = doc.getElementsByTagName("DICE")[0]
    sid = a.getAttribute("value")
    dice.append(float(sid))

dice=np.array(dice)
d=statistics.mean(dice)
print('Dsc:'+str(d))