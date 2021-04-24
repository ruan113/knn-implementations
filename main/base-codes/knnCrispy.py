#Passos para realizar o algoritmo knn:

#----------------------------Preparação dos dados----------------------------
# - Leitura do arquivo com os dados para treino do algoritmo.
# - No caso, usaremos uma "porcentagem" dos dados para uso como dados de treinamento e 
#   outra como dados para teste, assim teremos como checar como nosso algoritmo
#   se saiu ao final da classificação.  

#------------------------Para cada instancia de teste------------------------
# 
# -----Encontrando os vizinhos mais proximos de uma instancia de teste X-----
# - Encontre os vizinhos mais proximos do laço que estamos analisando, através do calculo
#   da distancia euclidiana entre dois pontos.
# - Neste calculo consideraremos a distancia com cada dado dentro da instancia analisada,
#   ou seja, cada atributo contará.
# Working in progress

# import the dependencies
import random
import math
import operator 
import sys
import time
from loadData import loadDataset
 
# this function is for calculating euclidan distance 
# between 2 points (every of train data and test data).
# this is a basic of pythagoras.
# params:
# - instance1: test instance that cotains all test data
# - instance2: 1 row of training data
# - length: length of features - 1 (for loop)
# output:
# - euclidean distance
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((float(instance1[x]) - float(instance2[x])), 2)
    return math.sqrt(distance)

# get testInstance nearest neighbors in trainingSet 
# params:
# - trainingSet: set of training data
# - testInstance: instance of test data that we'll going to find his nearest neighbors
# output:
# - k-neighbors data
def getNeighbors(trainingSet, testInstance, k):
    distances = []

    # get the lengh - 1, because the last one is not data, but definition
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        # calculate euclidean distance of test instance and training instance,
        # they have 'length' of columns
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        # collect training instance and it's distance to an array
        distances.append((trainingSet[x], dist))

    # sort the array of distance by it's distance's value
    distances.sort(key=operator.itemgetter(1))

    # get k-nearest neigbors
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# count the most nearest neighbors
# params:
# - neighbors: neigbors data
# output:
# - the definition, based on the most nearest neigbors
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    # sort classVotes by it's value descending, to get the maximum one
    sortedVotes = sorted(list(classVotes.items()), key=operator.itemgetter(1), reverse=True)
    # return the first one
    return sortedVotes[0][0]

# calculate the percentage of program's accuracy
# comparing the result of predictions to testSet
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

# main program
def run(filePath, split, numColunas, k):
    # get the start time
    startTime = time.time()

    trainingSet=[]
    testSet=[]

    # Preenchimento das instancias com os dados dentro do arquivo especificado
    loadDataset(filePath, split, trainingSet, testSet, numColunas)

    # print info
    print('Inicio (' + filePath + '):\n')
    print('Proporção para treinamento: ' + repr(split))
    print('Instancias de treinamento: ' + repr(len(trainingSet)))
    print('Instancias de teste: ' + repr(len(testSet)))
    
    predictions=[]

    for x in range(len(testSet)):
        # get neighbors as much as K
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        # determine the prediction
        result = getResponse(neighbors)
        # add the prediction into array
        predictions.append(result)

    # calculate the accuracy
    accuracy = getAccuracy(testSet, predictions)

    # print info
    print('Resultado:')
    print('Taxa de acertos: ' + repr(round(accuracy,2)) + '%')
    print('Tempo de execução: ' + str(round(time.time() - startTime, 2)) + ' seconds')
    print('\nProgram Ends')
    print('-----------------------------------------')
