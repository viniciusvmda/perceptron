#encoding: utf-8
'''----------------------------------------------------------------------
| Tarefa 8 - Implementar o perceptron para predizer o tipo do vinho     |
| 																		|
| Renan Mateus Bernardo Nascimento										|
| Vinícius Magalhães D'Assunção											|
----------------------------------------------------------------------'''

import numpy as np
import matplotlib.pyplot as plt
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
        self.erro = []                                                          # vetor de erros utilizado no plot
        self.tempo = []                                                         # vetor de tempo utilizado no plot


    '''
    ' Calcula a saída para a entrada x
    '''
    def calcular(self, x):
        return np.matmul(self.w, x)


    '''
    ' Calcula o erro a partir do erro quadrático médio
    '''
    def retornaErro(self, e):
        E = 0
        i = 0
        for i in range(0, e.shape[0]):
            Es = 0
            j = 0
            for j in range(0, e.shape[1]):
                Es += pow(e[i][j], 2)
            E += Es / j
        E = E / i
        return E
    

    '''
    ' Função de ativação do tipo degrau
    '''
    def f(self, u):
        y = np.zeros(shape=u.shape, dtype=np.int32)
        for i in range(0, u.shape[0]):
            if len(u.shape) > 1:
                for j in range(0, u.shape[1]):        
                    if u[i][j] >= 1:
                        y[i][j] = 1
                    else:
                        y[i][j] = 0
            else:
                if u[i] >= 1:
                    y[i] = 1
                else:
                    y[i] = 0
        return y


    '''
    ' Gera o gráfico Erro x Iteração
    '''
    def gerarGrafico(self):
        plt.plot(self.tempo, self.erro)
        plt.ylabel('Erro')
        plt.xlabel('Iteração')
        plt.title('Gráfico Erro x Iteração')
        plt.show()

    '''
    ' Treinamento supervisionado
    '''
    def treinar(self):
        self.tempo = []
        self.erro = []
        t = 1                                       # tempo
        E = 1                                       # erro
        e = np.zeros(shape=self.d.shape)            # vetor de erros
        while (t < self.max_it and E > 0):
            u = np.matmul(self.w, self.x) + self.b
            y = self.f(u)
            e = self.d - y
            self.w = self.w + np.matmul((self.alfa * e), np.transpose(self.x))
            self.b = self.b + self.alfa * e
            E = self.retornaErro(e)
            self.tempo.append(t)
            self.erro.append(E)
            t = t + 1