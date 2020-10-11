# Algorítmo genético:
from hard_cross_functions import tirar_o_ao, colocar_o_ao, inverter_a_lista, \
    teste_compartilhados, calcular_h_aneis, calcular_h_reservatorio, calcular_R,\
    calcular_delta_Q, somar_delta_Q, calculo_vazao_final, calcular_pressao, \
    print_vazao, print_pressao
import random
from pyeasyga import pyeasyga
import numpy as np

data = {"L_aneis": [np.array([164, 152, 168, 176, 224, 184, 124]),
                    np.array([124, 184, 208, 104, 200, 136, 228, 168]),
                    np.array([256, 264, 132, 320, 104, 168, 104, 208])],
        "D_aneis": [np.array([0.2, 0.15, 0.15, 0.1, 0.2, 0.15, 0.2]),
                    np.array([0.2, 0.15, 0.1, 0.15, 0.2, 0.15, 0.15, 0.2]),
                    np.array([0.15, 0.2, 0.15, 0.2, 0.1, 0.2, 0.15, 0.1])],
        "trechos_aneis": [np.array(["1 ao 9", "9 ao 10", "10 ao 11", "11 ao 12", "12 ao 3", "3 ao 2", "2 ao 1"]),
                          np.array(["1 ao 2", "2 ao 3", "3 ao 4", "4 ao 5", "5 ao 6", "6 ao 7", "7 ao 8", "8 ao 1"]),
                          np.array(["3 ao 13", "13 ao 14", "14 ao 15", "15 ao 16", "16 ao 17", "17 ao 5", "5 ao 4",
                                    "4 ao 3"])],
        "Q_reservatorio": np.array([4.4, 22, 26.4, 35.2, 17.6, 8.8, 30.8, 44, 48.4, 52.8, 39.6, 13.2]) / 1000,
        "P_6": np.array([9.27, 8.83, 8.64, 8.18, 8.99, 9.21, 8.42, 7.59, 7.25, 6.89, 7.9, 9.12]),
        "P_11": np.array([26.49, 26.25, 26.15, 25.9, 26.33, 26.45, 26.03, 25.59, 25.41, 25.22, 25.75, 26.4]),
        "P_15": np.array([20.57, 20.05, 19.83, 19.28, 20.23, 20.49, 19.57, 18.6, 18.22, 17.79, 18.96, 20.38]),
        "Q_base": [np.array([0, -0.069, -0.125, -0.18, -0.221, -0.888, -0.923]),
                   np.array([0.923, 0.888, 0, -0.029, 0.126, 0.077, 0.046, 0]),
                   np.array([0.612, 0.53, 0.509, 0.438, 0.321, 0.253, 0.029, 0])],
        "cota_nos": np.array([229, 209.5, 204.6, 199.4, 199.5, 198.5, 219.7, 207.5, 202.2, 197.8, 194.5, 202.5, 200.5,
                              204.5, 209.5, 208.4, 196.5, 201.6]),
        "L_reservatorio": 324,
        "D_reservatorio": 0.3}

"""Há uma informação de que o anel que contém o nó 8 foi o primeiro a ser construído, seguido do que contém o nó 9 e por
último o que contém o 14. Porém, múltiplas mudanças têm ocorrido na rede, e você deverá avaliar a melhor forma de usar
(ou não) essa informação."""


def create_individual(data):
    anel_1 = random.uniform(0.00001, 0.001)
    anel_2 = random.uniform(0.00001, 0.001)
    anel_3 = random.uniform(0.00001, 0.001)
    while anel_1 <= anel_2 or anel_1 <= anel_3 or anel_2 <= anel_3:
        anel_1 = random.uniform(0.00001, 0.001)
        anel_2 = random.uniform(0.00001, 0.001)
        anel_3 = random.uniform(0.00001, 0.001)
    individuo = [anel_1, anel_2, anel_3]
    return (individuo)


def mutate(individual):
    mutation_index = random.randrange(len(individual))
    if mutation_index == 0:
        individual[mutation_index] = random.uniform(individual[1], 0.001)
    elif mutation_index == 1:
        individual[mutation_index] = random.uniform(individual[2], individual[0])
    elif mutation_index == 2:
        individual[mutation_index] = random.uniform(0.00001, individual[1])
    print("mutation: ", individual)


def cross_over(parent_1, parent_2):
    child_1 = [parent_1[0], parent_1[1], parent_2[2]]
    child_2 = [parent_1[0], parent_2[1], parent_2[2]]
    print(child_1)
    print(child_2)
    return (child_1, child_2)


def fitness(individual, data):
    print(individual)
    erro_pressao = 0
    for i in range(len(data["Q_reservatorio"])):
        Q_aneis = np.array([data["Q_base"][0] * data["Q_reservatorio"][i],
                            data["Q_base"][1] * data["Q_reservatorio"][i],
                            data["Q_base"][2] * data["Q_reservatorio"][i]])
        r_anel_1 = np.array([individual[1], individual[1], individual[1], individual[1], individual[1], individual[0],
                             individual[0]])
        r_anel_2 = np.array([individual[0]] * 8)
        r_anel_3 = np.array([individual[2], individual[2], individual[2], individual[2], individual[2], individual[2],
                             individual[0], individual[0]])
        rugosidade_aneis = np.array([r_anel_1, r_anel_2, r_anel_3])
        Q_final = np.array(calculo_vazao_final(data["trechos_aneis"],
                                               data["L_aneis"],
                                               data["D_aneis"],
                                               Q_aneis,
                                               rugosidade_aneis))
        pressao = calcular_pressao(data["trechos_aneis"],
                                   data["cota_nos"],
                                   data["Q_reservatorio"][i],
                                   data["L_reservatorio"],
                                   data["D_reservatorio"],
                                   Q_final,
                                   data["L_aneis"],
                                   data["D_aneis"],
                                   rugosidade_aneis,
                                   individual[0])
        erro_pressao += abs(pressao[6] - data["P_6"][i])
        erro_pressao += abs(pressao[11] - data["P_11"][i])
        erro_pressao += abs(pressao[15] - data["P_15"][i])

    print(erro_pressao, "\n")
    return (erro_pressao)


ga = pyeasyga.GeneticAlgorithm(data,
                               population_size=50,
                               generations=50,
                               mutation_probability=0.5,
                               crossover_probability=0.9,
                               maximise_fitness=False,
                               elitism=True)

ga.create_individual = create_individual
ga.mutate_function = mutate
ga.crossover_function = cross_over
ga.fitness_function = fitness
ga.run()
print(ga.best_individual())

# A PARTIR DAQUI EU USEI A RUGOSIDADE OBTIDA COM O AG PARA COMPARAR AS PRESSOES
def comparar_pressoes(r_obtido_ga, data):
    r_anel_1 = np.array([r_obtido_ga[1], r_obtido_ga[1], r_obtido_ga[1], r_obtido_ga[1], r_obtido_ga[1],
                         r_obtido_ga[0], r_obtido_ga[0]])
    r_anel_2 = np.array([r_obtido_ga[0]] * 8)
    r_anel_3 = np.array([r_obtido_ga[2], r_obtido_ga[2], r_obtido_ga[2], r_obtido_ga[2], r_obtido_ga[2],
                         r_obtido_ga[2], r_obtido_ga[0], r_obtido_ga[0]])
    rugosidade_aneis = np.array([r_anel_1, r_anel_2, r_anel_3])

    for i in range(len(data["Q_reservatorio"])):
        Q_aneis = np.array([data["Q_base"][0] * data["Q_reservatorio"][i],
                            data["Q_base"][1] * data["Q_reservatorio"][i],
                            data["Q_base"][2] * data["Q_reservatorio"][i]])
        Q_final = np.array(calculo_vazao_final(data["trechos_aneis"], data["L_aneis"], data["D_aneis"], Q_aneis,
                                               rugosidade_aneis))
        pressao = calcular_pressao(data["trechos_aneis"], data["cota_nos"], data["Q_reservatorio"][i], 324, 0.3,
                                   Q_final,
                                   data["L_aneis"], data["D_aneis"], rugosidade_aneis, r_obtido_ga[0])
        print("Vazão: " +  str(data["Q_reservatorio"][i] * 1000), " L/s")
        print(pressao[6])
        print(pressao[11])
        print(pressao[15], "\n")
        print(abs(pressao[6] - data["P_6"][i]))
        print(abs(pressao[11] - data["P_11"][i]))
        print(abs(pressao[15] - data["P_15"][i]), "\n")

#rodar 10x o algoritmo genético
rugosidades_ga = []
for i in range(10):
    ga = pyeasyga.GeneticAlgorithm(data,
                                   population_size=50,
                                   generations=50,
                                   mutation_probability=0.5,
                                   crossover_probability=0.9,
                                   maximise_fitness=False,
                                   elitism=True)
    ga.create_individual = create_individual
    ga.mutate_function = mutate
    ga.crossover_function = cross_over
    ga.fitness_function = fitness
    ga.run()
    rugosidades_ga.append(ga.best_individual())


best_r_obtido_ga = [0.00011825568106607827, 9.143566722726664e-05, 1.763385820534409e-05]
comparar_pressoes(best_r_obtido_ga, data)


# for i in range(len(data["Q_reservatorio"])):
#     print("Vazão: ", data["Q_reservatorio"][i] * 1000, " L/s")
#     consumo_nos = [0., 0.000847, 0.000385, 0.000605, 0.000319, 0.001078, 0.000539, 0.000341, 0.000506, 0.000759,
#                    0.000616, 0.000605, 0.000451, 0.000902, 0.000231, 0.000781, 0.001287, 0.000748]
#     Q_aneis = np.array([data["Q_base"][0] * data["Q_reservatorio"][i],
#                         data["Q_base"][1] * data["Q_reservatorio"][i],
#                         data["Q_base"][2] * data["Q_reservatorio"][i]])
#     Q_final = np.array(calculo_vazao_final(trechos_aneis, L_aneis, D_aneis, Q_aneis, rugosidade_aneis))
#
#     consumo_nos_2 = ((np.array(consumo_nos) / (11 / 1000)) * data["Q_reservatorio"][i]) * 1000
#     for j in enumerate(consumo_nos_2):
#         print (j)
