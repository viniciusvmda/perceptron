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
    def __init__(self, entrada, teste, taxa_aprendizagem, max_it):
        self.x = entrada                                # entrada
        self.linhas, self.colunas = entrada.shape
        self.w = np.zeros(shape=(self.linhas))          # vetor de pesos
        self.b = np.zeros(shape=(self.colunas))         # bias
        self.d = teste                                  # vetor de teste
        self.alfa = taxa_aprendizagem                   # taxa de aprendizagem
        self.max_it = max_it                            # número máximo de iterações


    '''
    ' Calcula a saída para a entrada x
    '''
    def calcular(self, x):
        return  np.matmul(self.w, x) + self.b


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
        e = np.zeros(shape=(self.linhas))           # vetor de erros
        while (t < self.max_it and E > 0):
            y = self.calcular(self.x)
            e = self.d - y
            self.w = self.w + np.matmul((self.alfa * e), np.transpose(self.x))
            self.b = self.b + self.alfa * e
            E = self.retornaErro(e)
            t = t + 1