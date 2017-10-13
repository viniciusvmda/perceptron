import numpy as np

arq = open('wine.data', 'r')
saidas = []
entradas = []

for linha in arq:
    linha = linha[:-1]
    linha = linha.split(',')

    saidas.append(int(linha[0]))
    entradas.append([ float(i) for i in linha[1:] ])


entradas = np.array(entradas)
saidas = np.array(saidas)

entradas = entradas / entradas.max(axis=0)

arq.close()

arq = open('wine.data', 'w')
for i in range(0, len(entradas)):
    entrada = ''
    saida = ''
    if saidas[i] == 1:
        saida = '1,0,0'
    elif saidas[i] == 2:
        saida = '0,1,0'
    elif saidas[i] == 3:
        saida = '0,0,1'
    for el in entradas[i]:
        entrada += ',' + str(el)
    arq.write(saida + entrada + '\n')

arq.close()