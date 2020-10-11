def tirar_o_ao(lista_de_aneis):
    numero_de_aneis = len(lista_de_aneis)
    ao = [[] for _ in range(numero_de_aneis)]
    for i in range(len(lista_de_aneis)):
        for j in range(len(lista_de_aneis[i])):
            ao[i].append(lista_de_aneis[i][j].split(" ao "))
    return (ao)


def colocar_o_ao(lista_de_aneis_sem_o_ao):
    numero_de_aneis = len(lista_de_aneis_sem_o_ao)
    ao = [[] for _ in range(numero_de_aneis)]
    for i in range(len(lista_de_aneis_sem_o_ao)):
        for j in range(len(lista_de_aneis_sem_o_ao[i])):
            ao[i].append(lista_de_aneis_sem_o_ao[i][j][0] + " ao " + lista_de_aneis_sem_o_ao[i][j][1])
    return (ao)


def inverter_a_lista(lista):
    import numpy as np
    numero_aneis = len(lista)
    invertida = [[] for _ in range(numero_aneis)]
    for i in range(len(lista)):
        for j in range(len(lista[i])):
            invertida[i].append(lista[i][j][::-1])
    invertida = np.array(invertida)
    return (invertida)


def teste_compartilhados(lista_de_aneis):
    import numpy as np
    a = tirar_o_ao(lista_de_aneis)
    numero_de_aneis = len(lista_de_aneis)
    lista = [[] for _ in range(numero_de_aneis)]
    for x in range(len(a)):
        for y in range(len(a[x])):
            for z in range(len(a)):
                if a[x][y][::-1] in a[z]:
                    lista[x].append(a[x][y])
    lista = np.array(colocar_o_ao(lista))
    return (lista)

def calcular_h_aneis(Q_aneis, L_aneis, D_aneis, rugosidade_aneis, viscosidade_cinematica=10 ** -6):
    import math
    import numpy as np
    numero_de_aneis = len(Q_aneis)
    h_aneis = [[] for _ in range(numero_de_aneis)]
    for i in range(numero_de_aneis):
        for j in range(len(Q_aneis[i])):
            reynolds = ((4 * abs(Q_aneis[i][j])) / (math.pi * D_aneis[i][j] * viscosidade_cinematica))
            if reynolds == 0:
                h_aneis[i].append(0)
            else:
                dentro_log = (rugosidade_aneis[i][j] / (3.7 * D_aneis[i][j])) + (5.74 / (reynolds ** 0.9))
                f = 0.25 / ((math.log(dentro_log, 10)) ** 2)
                h_aneis[i].append((0.0827 * f * (Q_aneis[i][j] ** 2) * L_aneis[i][j] / (D_aneis[i][j] ** 5)))
            if Q_aneis[i][j] < 0:
                h_aneis[i][j] = h_aneis[i][j] * -1
    h_aneis = np.array(h_aneis)
    return (h_aneis)

# calcular perda de carga do trecho do reservatório (gambiarra)
def calcular_h_reservatorio(Q, L, D, rugosidade_reservatorio, viscosidade_cinematica=10 ** -6):
    import math
    reynolds = ((4 * abs(Q)) / (math.pi * D * viscosidade_cinematica))
    dentro_log = (rugosidade_reservatorio / (3.7 * D)) + (5.74 / (reynolds ** 0.9))
    f = 0.25 / ((math.log(dentro_log, 10)) ** 2)
    h = ((0.0827 * f * (Q ** 2) * L / (D ** 5)))
    return h


# calcular o fator para correção de delta Q
def calcular_R(Q_aneis, h_aneis):
    numero_de_aneis = len(Q_aneis)
    R = [[] for _ in range(numero_de_aneis)]
    for i in range(numero_de_aneis):
        for j in range(len(Q_aneis[i])):
            if Q_aneis[i][j] == 0:
                R[i].append(0)
            else:
                R[i].append((2 * h_aneis[i][j]) / Q_aneis[i][j])
    return (R)


# calcular o delta Q sem considerar os trechos compartilhados
def calcular_delta_Q(h_aneis, R_aneis):
    import numpy as np
    numero_aneis = len(h_aneis)
    delta_Q = [[] for _ in range(numero_aneis)]
    for i in range(len(h_aneis)):
        h = np.sum(h_aneis[i])
        R = np.sum(R_aneis[i])
        delta_Q[i].append(- h / R)
    return (delta_Q)


# calculo do delta Q, considerando os trechos compartilhados e corrigindo as vazões
def somar_delta_Q(delta_Q, Q_aneis, trechos_compartilhados, trechos_aneis):
    import numpy as np
    compartilhados = tirar_o_ao(trechos_compartilhados)
    compartilhados_invertido = inverter_a_lista(tirar_o_ao(trechos_compartilhados))
    trechos = np.array(tirar_o_ao(trechos_aneis))
    numero_aneis = len(delta_Q)
    novo_Q = [[] for _ in range(numero_aneis)]

    for i in range(numero_aneis):
        for j in range(len(Q_aneis[i])):
            novo_Q[i].append(delta_Q[i][0] + Q_aneis[i][j])

    novo_Q = np.array(novo_Q)

    for i in range(len(compartilhados_invertido)):
        for j in range(len(compartilhados_invertido[i])):
            for k in range(len(trechos)):
                if compartilhados_invertido[i][j] in trechos[k]:
                    delta = - delta_Q[k][0]
                    indice = list(trechos[i]).index(compartilhados[i][j])
                    novo_Q[i][indice] += delta

    return (novo_Q)


# calculo da vazão após as iterações com limite na perda de carga
def calculo_vazao_final(trechos_aneis, L_aneis, D_aneis, Q_aneis, rugosidade_aneis, limite_h=0.000001):
    import numpy as np
    trechos_compartilhados = teste_compartilhados(trechos_aneis)
    h_aneis = calcular_h_aneis(Q_aneis, L_aneis, D_aneis, rugosidade_aneis)
    soma_de_h = []

    for i in range(len(h_aneis)):
        soma_de_h.append(np.sum(h_aneis[i]))

    while True in (abs(x) > limite_h for x in soma_de_h):
        R_aneis = calcular_R(Q_aneis, h_aneis)
        delta_Q = calcular_delta_Q(h_aneis, R_aneis)
        Q_aneis = somar_delta_Q(delta_Q, Q_aneis, trechos_compartilhados, trechos_aneis)
        h_aneis = calcular_h_aneis(Q_aneis, L_aneis, D_aneis, rugosidade_aneis)
        for j in range(len(h_aneis)):
            soma_de_h[j] = np.sum(h_aneis[j])
    return (Q_aneis)


# função para calcular as pressões nos nós
def calcular_pressao(trechos_aneis, cota_nos, Q_reservatorio, L_reservatorio, D_reservatorio, Q_final, L_aneis, D_aneis,
                     rugosidade_aneis, rugosidade_reservatorio):
    h_reservatorio = calcular_h_reservatorio(Q_reservatorio, L_reservatorio, D_reservatorio, rugosidade_reservatorio)
    h_final = calcular_h_aneis(Q_final, L_aneis, D_aneis, rugosidade_aneis)
    pressao = [0] * len(cota_nos)
    pressao[1] = pressao[0] + cota_nos[0] - cota_nos[1] - h_reservatorio
    trechos = tirar_o_ao(trechos_aneis)
    for i in range(len(trechos)):
        for j in range(len(trechos[i])):
            indice1 = int(trechos[i][j][1])
            indice2 = int(trechos[i][j][0])
            pressao[indice1] = pressao[indice2] + cota_nos[indice2] - cota_nos[indice1] - h_final[i][j]
    return (pressao)


# trecho de print:
def print_vazao(Q_final, trechos_aneis):
    print("Vazões: ")
    for i in range(len(Q_final)):
        print("\n")
        print("Anel ", i + 1, ": ")
        for j in range(len(Q_final[i])):
            print(trechos_aneis[i][j], " : ", Q_final[i][j] * 1000)


def print_pressao(pressao):
    print("Carga de Pressão: ", "\n")
    for i in range(len(pressao)):
        print("Nó ", i, ":", pressao[i])
