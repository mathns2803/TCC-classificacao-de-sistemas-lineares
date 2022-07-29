# -*- coding: utf-8 -*-
"""
Created on Sun May 15 14:13:37 2022

@author: Matheus Henrique Nunes Silveira
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

"Dados para treino"
ordem1 = cm.readTxtFile(".\\dados\\ordem1.txt")
ordem1 = ordem1[:][700::]
ordem2 = cm.readTxtFile(".\\dados\\ordem2.txt")
ordem2 = ordem2[:][700::]
aOrdem1 = cm.readTxtFile(".\\dados\\ordem1Atraso.txt")
aOrdem1 = aOrdem1[:][700::]
aOrdem2 = cm.readTxtFile(".\\dados\\ordem2Atraso.txt")
aOrdem2 = aOrdem2[:][700::]
satura = cm.readTxtFile(".\\dados\\ampOpInteg.txt")
satura = satura[:][700::]

"Dados para teste"
testeOrdem1 = cm.readTxtFile(".\\dados\\TesteO1.txt")
testeOrdem1 = testeOrdem1[:][700::]
testeAOrdem1 = cm.readTxtFile(".\\dados\\TesteO1A.txt")
testeAOrdem1 = testeAOrdem1[:][700::]
testeOrdem2 = cm.readTxtFile(".\\dados\\TesteO2.txt")
testeOrdem2 = testeOrdem2[:][700::]
testeAOrdem2 = cm.readTxtFile(".\\dados\\TesteO2A.txt")
testeAOrdem2 = testeAOrdem2[:][700::]
testeSatu = cm.readTxtFile(".\\dados\\TesteAmp.txt")
testeSatu = testeSatu[:][700::]


"""
Manipulação dos dados para que seja possível a realização do treino, nesta parte é adicionado todos os dados
em um array de traino e outro para teste, bem como suas respostas são adicionadas em um outro array de lables.

O módulo aplicado proporciona uma quebra de dados em 8 partes, onde as 4 primeiras colunas representam a entrada
e as 4 últimos a saída dada um fator de intervalo para que seja possível identificar a transição dos dados.

Lables e suas definições:
    0 - Primeira Ordem
    1 - Segunda Ordem 
    2 - Primeira Ordem com Atraso puro de tempo
    3 - Segunda Ordem com Atraso puro de tempo
    4 - Saturação
"""

"Este fator determina qual o intervalo entre 4 pontos de entrada"
factor = 10

"Inicio do array de treino"
treino = []
train_lables = []

"Adicionando os dados de treino em seu array e seus grupos"
ordem1 = cm.divideInputIn8Parts(ordem1, factor)
for i in np.transpose(ordem1):
    treino.append(i)
    train_lables.append(0)
ordem2 = cm.divideInputIn8Parts(ordem2, factor)
for i in np.transpose(ordem2):
    treino.append(i)
    train_lables.append(1)
# aOrdem1 = cm.divideInputIn8Parts(aOrdem1, factor)
# for i in np.transpose(aOrdem1):
#     treino.append(i)
#     train_lables.append(2)
# aOrdem2 = cm.divideInputIn8Parts(aOrdem2, factor)
# for i in np.transpose(aOrdem2):
#     treino.append(i)
#     train_lables.append(3)
# satura = cm.divideInputIn8Parts(satura, factor)
# for i in np.transpose(satura):
#     treino.append(i)
#     train_lables.append(2)

"Inicio do array de teste"
test = []
test_lables = []

"Adicionando os dados de teste em seu array e seus grupos"
testeOrdem1 = cm.divideInputIn8Parts(testeOrdem1, factor)
for i in np.transpose(testeOrdem1):   
    test.append(i)
    test_lables.append(0)
testeOrdem2 = cm.divideInputIn8Parts(testeOrdem2, factor)
for i in np.transpose(testeOrdem2):
    test.append(i)
    test_lables.append(1)
# testeAOrdem1 = cm.divideInputIn8Parts(testeAOrdem1, factor)
# for i in np.transpose(testeAOrdem1):
#     test.append(i)
#     test_lables.append(2)
# testeAOrdem2 = cm.divideInputIn8Parts(testeAOrdem2, factor)
# for i in np.transpose(testeAOrdem2):
#     test.append(i)
#     test_lables.append(3)
# testeSatu = cm.divideInputIn8Parts(testeSatu, factor)
# for i in np.transpose(testeSatu):
#     test.append(i)
#     test_lables.append(2)

"Início do treino - Montando a rede neural"
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8), #Quantidade de neuronios de entrada
    tf.keras.layers.Dense(128, activation='relu'), #Quantidade de neuronios de saída
    # tf.keras.layers.Dense(2, activation='softmax') # Última camada com as 2 saida
    # tf.keras.layers.Dense(3, activation='softmax') # Última camada com as 3 saida
    tf.keras.layers.Dense(2, activation='softmax') # Última camada com as 5 saida
])

model.compile(optimizer='adam', #Sistema de otimização dos dados
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"Quantidade de épocas"
epo = 30

'Executando o modelo com os dados'
history = model.fit(np.array(treino),
                    np.array(train_lables),
                    validation_data=(np.array(test), np.array(test_lables)),
                    epochs=epo)  # Treino Aqui devo fazer um filro de melhor ponto de epoca
# test_loss, test_acc = model.evaluate(np.array(test), np.array(test_lables), verbose=5)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epo)


'Inserir no gráfico a acurácia e perdas'
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Taxa de acerto no treino')
plt.plot(epochs_range, val_acc, label='Taxa de acerto na validação', color='green', marker='o')
plt.legend(loc='lower right')
plt.title('Taxa de acerto  no treino e na validação')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perda no treino')
plt.plot(epochs_range, val_loss, label='Perda na validação', color='green', marker='o')
plt.legend(loc='upper right')
plt.title('Perda no treino e na validação')
plt.show()

cmr = []

"Neste ponto é realizados testes um a um e asim montado a matriz de confusão"
x = np.transpose(cm.divideInputIn8Parts(testeOrdem1, 50))
sing = 0
result1 =[0,0]
for i in np.transpose(testeOrdem1):
    sing = sing + 1
    predictions_single = model.predict(np.expand_dims(i,0))
    result1[0] = result1[0] + predictions_single[0][0]
    result1[1] = result1[1] + predictions_single[0][1]
    # result1[2] = result1[2] + predictions_single[0][2]
    # result1[3] = result1[3] + predictions_single[0][3]
    # result1[4] = result1[4] + predictions_single[0][4]
    cmr.append(predictions_single)
x = np.transpose(cm.divideInputIn8Parts(testeOrdem2, 50))
sing = 0
result2 =[0,0]
for i in np.transpose(testeOrdem2):
    sing = sing + 1
    # print(sing)
    predictions_single = model.predict(np.expand_dims(i,0))
    result2[0] = result2[0] + predictions_single[0][0]
    result2[1] = result2[1] + predictions_single[0][1]
    # result2[2] = result2[2] + predictions_single[0][2]
    # result2[3] = result2[3] + predictions_single[0][3]
    # result2[4] = result2[4] + predictions_single[0][4]
    cmr.append(predictions_single)
# x = np.transpose(cm.divideInputIn8Parts(testeAOrdem1, 50))
# sing = 0
# result3 =[0,0,0,0,0]
# for i in np.transpose(testeAOrdem1):
#     sing = sing + 1
#     predictions_single = model.predict(np.expand_dims(i,0))
#     result3[0] = result3[0] + predictions_single[0][0]
#     result3[1] = result3[1] + predictions_single[0][1]
#     result3[2] = result3[2] + predictions_single[0][2]
#     result3[3] = result3[3] + predictions_single[0][3]
#     result3[4] = result3[4] + predictions_single[0][4]
#     cmr.append(predictions_single)
# x = np.transpose(cm.divideInputIn8Parts(testeAOrdem2, 50))
# sing = 0
# result4 =[0,0,0,0,0]
# for i in np.transpose(testeAOrdem2):
#     sing = sing + 1
#     # print(sing)
#     predictions_single = model.predict(np.expand_dims(i,0))
#     result4[0] = result4[0] + predictions_single[0][0]
#     result4[1] = result4[1] + predictions_single[0][1]
#     result4[2] = result4[2] + predictions_single[0][2]
#     result4[3] = result4[3] + predictions_single[0][3]
#     result4[4] = result4[4] + predictions_single[0][4]
#     cmr.append(predictions_single) 
# x = np.transpose(cm.divideInputIn8Parts(testeSatu, 50))
# sing = 0
# result5 =[0,0,0]
# for i in np.transpose(testeSatu):
#     sing = sing + 1
#     predictions_single = model.predict(np.expand_dims(i,0))
#     result5[0] = result5[0] + predictions_single[0][0]
#     result5[1] = result5[1] + predictions_single[0][1]
#     result5[2] = result5[2] + predictions_single[0][2]
#     result5[3] = result5[3] + predictions_single[0][3]
#     result5[4] = result5[4] + predictions_single[0][4]
#     cmr.append(predictions_single)

'Resultados em uma matriz de confusão'
result = []
result.append(result1)
result.append(result2)
# result.append(result3)
# result.append(result4)
# result.append(result5)

for r in result:
    print(r)
t = []
for cc in cmr:
    for c in cc:
        if float(c[0]) > float(c[1]):
            # if float(c[0]) > float(c[2]):
                # if float(c[0]) > float(c[3]):
                #     if float(c[0]) > float(c[4]):
            t.append(0)
                    
        if float(c[1]) > float(c[0]):
            # if float(c[1]) > float(c[2]):
                # if float(c[1]) > float(c[3]):
                #     if float(c[1]) > float(c[4]):
            t.append(1)
                        
        # if float(c[2]) > float(c[0]):
        #     if float(c[2]) > float(c[1]):
        #         if float(c[2]) > float(c[3]):
        #             if float(c[2]) > float(c[4]):
        #                 t.append(2)
        # if float(c[3]) > float(c[0]):
        #     if float(c[3]) > float(c[1]):
        #         if float(c[3]) > float(c[2]):
        #             if float(c[3]) > float(c[4]):
        #                     t.append(3)
        # if float(c[4]) > float(c[0]):
        #     if float(c[4]) > float(c[1]):
        #         if float(c[4]) > float(c[2]):
        #             if float(c[4]) > float(c[3]):
        #                     t.append(4)

cm = confusion_matrix(train_lables, t, normalize='all')
cm = cm*cm.shape[0]
ax = sns.heatmap(cm, annot=True, vmin=0, vmax=1,
            fmt='.2%', cmap='Blues')
ax.set_title('Matriz de confusão');
ax.set_xlabel('\nRespostas previstas')
ax.set_ylabel('Respostas esperadas ');
## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(['O. 1','O. 2', 'O. 1 Atr','O. 2 Atr','Saturação'])
# ax.yaxis.set_ticklabels(['O. 1','O. 2', 'O. 1 Atr','O. 2 Atr', 'Saturação'])
ax.xaxis.set_ticklabels(['O. 1','O. 2'])
ax.yaxis.set_ticklabels(['O. 1','O. 2'])
## Display the visualization of the Confusion Matrix.
plt.show()
    
fim = time.time()
print(fim - inicio)