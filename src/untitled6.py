# -*- coding: utf-8 -*-
"""
"""
import tensorflow as tf
import commons.helppers as cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time

inicio = time.time()

nomesDasClasses = ['PrimeiraOrdem', 'PrimeiraOrdemComAtrasoPuroDeTempo', 'SegundaOrdem', 'SegundaOrdemComAtrasoPuroDeTempo', 'SistemaComSaturacao'  ]

"""
Inicio do import dos dados dentro do sistema
"""
# fonte = cm.readTxtFile(".\\dados\\fonte.txt")
"Dados para treino"
ordem1 = cm.readTxtFile(".\\dados\\TesteO2A.txt")
# ordem1 = cm.divideInputIn8Parts(ordem1, 50)

# plt.figure(1)
# plt.plot(fonte[0])
# ordem1.drop()
# ordem1.concat(fonte[0])
plt.figure(2)
plt.plot(ordem1[0])