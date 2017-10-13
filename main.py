#encoding: utf-8
'''----------------------------------------------------------------------
| Tarefa 8 - Implementar o perceptron para predizer o tipo do vinho     |
| 																		|
| Renan Mateus Bernardo Nascimento										|
| Vinícius Magalhães D'Assunção											|
----------------------------------------------------------------------'''

import sys
import numpy as np
from perceptron import Perceptron


'''
' Lê o arquivo e separa as linhas em vetores de entrada e saída de treino e teste
'''
def leArquivo(nome_arquivo):
    arq = open(nome_arq, 'r')
    entradas = []
    saidas = []
    for linha in arq:
        # Remove o \n
        linha = linha[:-1]
        # Separa a entrada pela vírgula
        aux = linha.split(',')
        saidas.append(int(aux[0]))
        entradas.append([ float(i) for i in aux[1:] ])
        
    # Define a quantidade de entradas de treino e de teste
    tamanho     = len(entradas)
    qtd_treino  = int(2/3 * tamanho)

    # Separa os vetores de treino
    treino          = np.array(entradas[:qtd_treino])
    treino          = np.transpose(treino)
    saida_treino    = np.array(saidas[:qtd_treino])

    # Separa os vetores de teste
    teste           = np.array(entradas[qtd_treino:])
    teste           = np.transpose(teste)
    saida_teste     = np.array(saidas[qtd_treino:])

    return treino, saida_treino, teste, saida_teste


'''
' Main
'''
# Verifica se o usuário passou os parâmetro
num_args = len(sys.argv)
if num_args < 4:
    sys.exit("Faltam argumentos: python3 main.py <nome_arquivo> <taxa_aprendizagem> <max_it>")

nome_arq = sys.argv[1]
taxa_aprendizagem = int(sys.argv[2])
max_it = int(sys.argv[3])

treino, saida_treino, teste, saida_teste = leArquivo(nome_arq)

# construtor (entrada, teste, taxa_aprendizagem, max_it)
P = Perceptron(treino, saida_treino, taxa_aprendizagem, max_it)
print("Treinando rede...")
P.treinar()
print("Rede treinada com sucesso")

linhas, colunas = teste.shape
print("\nInício dos testes\n")
for j in range(0, colunas):
    entrada = [ teste[i][j] for i in range(0, linhas) ]
    saida = P.calcular(entrada)
    print("Entrada: " + str(entrada))
    print("Saída: " + str(saida))
    print("Saída desejada: " + str(saida_teste[j]) + '\n')
