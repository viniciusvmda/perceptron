#encoding: utf-8
'''----------------------------------------------------------------------
| Tarefa 8 - Implementar o perceptron para predizer o tipo do vinho     |
| 																		|
| Renan Mateus Bernardo Nascimento										|
| Vinícius Magalhães D'Assunção											|
----------------------------------------------------------------------'''

import numpy as np

class Perceptron:
    
    '''
    ' Construtor
    '''
    def __init__(self, entrada, saida, taxa_aprendizagem, max_it):
        LINHA = 0
        COLUNA = 1
        self.x = entrada                                                        # entrada
        self.linhas, self.colunas = entrada.shape
        self.w = np.zeros(shape=(saida.shape[LINHA], entrada.shape[LINHA]))     # vetor de pesos
        self.d = saida                                                          # vetor de saida desejada
        self.b = np.zeros(shape=self.d.shape)                                   # bias
        self.alfa = taxa_aprendizagem                                           # taxa de aprendizagem
        self.max_it = max_it                                                    # número máximo de iterações


    '''
    ' Calcula a saída para a entrada x
    '''
    def calcular(self, x):
        return np.matmul(self.w, x)


    '''
    ' Calcukla o erro a partir da soma dos quadrados dos erros
    '''
    def retornaErro(self, e):
        E = 0
        for it in e:
            E += pow(it, 2)
        return E


    '''
    ' Treinamento supervisionado
    '''
    def treinar(self):
        t = 1                                       # tempo
        E = 1                                       # erro
        e = np.zeros(shape=self.d.shape)           # vetor de erros
        while (t < self.max_it):
            print('\n\nw: ' + str(self.w))
            print('\nx: ' + str(self.x))
            print('\nb: ' + str(self.b))
            y = np.matmul(self.w, self.x) + self.b
            print('\ny: ' + str(y))
            print('\nd: ' + str(self.d))
            e = self.d - y
            print('\ne: ' + str(e))
            print('alfa: ' + str(self.alfa))
            self.w = self.w + np.matmul((self.alfa * e), np.transpose(self.x))
            self.b = self.b + self.alfa * e
            E = self.retornaErro(e)
            t = t + 1