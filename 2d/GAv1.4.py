import random
import matplotlib.pyplot as plt
from crossover import *




#Genetic Algorithm Parameters
popSize = 300
elitismRate = 0.2
mutationRate = .05
crossOverRate = 0.8
generationNo = 200




fileName ="dataSets/fri26_d.txt"
chromosomeRollNo = 0
plotFlag = 1 # Assign 1 for plotting, 0 for not plotting


def readTSPData(fileName):
  sourceFile = open(fileName, "r")
  rawData = sourceFile.read()
  sourceFile.close()
  formattedData = []

  temp = ""
  tempLine = []
  for i in rawData:
    if i != " ":
      temp += str(i)
    if i == " " or i == "\n":
      if temp != "":
        temp = float(temp)
        tempLine.append(temp)
        temp = ""
    if i == "\n":
      formattedData.append(tempLine)
      tempLine = []
  temp = float(temp)
  tempLine.append(temp)
  formattedData.append(tempLine)
  return formattedData


dataMatrix = readTSPData(fileName)
cityList = [ i for i in range(0,len(dataMatrix)) ] #City List


class chromosomes:
  def __init__(self,route,chromosomeRollNo, parents=["NA","NA"]):
    self.chromosomeRollNo = chromosomeRollNo
    self.route = route
    self.distance = 0.0
    self.fitnessScore = 0.0
    self.parents = parents
    
  def __repr__(self):
    #return " " + str(self.chromosomeRollNo) + ") " + str(self.route) + " Route Distance = "+ str(self.distance) + " Fitness = " + str(self.fitnessScore) + "\n"
    return " " + str(self.chromosomeRollNo) + ") " + str(self.route) + " Cost = " + str(self.distance) + " Parents= "+ str(self.parents) + "\n"


def generateInitialPopulation(popSize, initialPopulation, cityList):
  global chromosomeRollNo  
  count = popSize

  while count > 0:
    chromosome = chromosomes(random.sample(cityList, len(cityList)),chromosomeRollNo)
    if chromosome not in initialPopulation:
      chromosomeRollNo += 1
      initialPopulation.append(chromosome)
      count -= 1
    

def fitnessOperator(chromosome):
  route = chromosome.route
  totalDistance = 0
  fromCity = 0
  toCity = 0
  for i in range(0,len(route)-1):
    fromCity = int(route[i])
    toCity = int(route[i+1])
    totalDistance += dataMatrix[fromCity][toCity]
  fromCity = toCity
  toCity = int(route[0])
  totalDistance += dataMatrix[fromCity][toCity]
  return totalDistance


def assignFitness(population):
  for i in population:
    i.distance = fitnessOperator(i)
    i.fitnessScore = 1/i.distance


def elitism(population, eliteChromosomes, elitismRate):
  eliteSize = int(len(population) * elitismRate)
  sortedPopulation =  sorted(population, key= lambda x : x.fitnessScore, reverse = True)
  for i in range(0,eliteSize):
    eliteChromosomes.append(sortedPopulation[i])


def rwSelection(population):
  totalFitness = 0
  for i in population:
    totalFitness += i.fitnessScore
  P = random.random()
  N = 0.0
  for i in population:
    N += i.fitnessScore/totalFitness
    if (N>P):
      return i


def selectParents(population, matingPool, eliteChromosomes, numberOfParents):
  count = numberOfParents

  while count > 0:
    selectedParent = rwSelection(population)
    if selectedParent not in matingPool:
      matingPool.append(selectedParent)
      count -= 1 

    
def orderedCrossOver(parent1, parent2):
  
  randomPoint1 = random.randint(0,(len(parent1) - 1))
  randomPoint2 = random.randint(0,(len(parent1) - 0))
  
  startGene = min(randomPoint1, randomPoint2)
  endGene = max(randomPoint1, randomPoint2)
  
  #child 1
  child1 = []
  parent1subset = []
  parent2subset = []
  
  for i in range(startGene, endGene):
    parent1subset.append(parent1[i])
  
  parent2subset = [item for item in parent2 if item not in parent1subset]
  child1 = parent1subset + parent2subset


  #child 2
  child2 = []
  parent1subset = []
  parent2subset = []

  for i in range(startGene, endGene):
    parent2subset.append(parent2[i])
  
  parent1subset = [item for item in parent1 if item not in parent2subset]
  child2 = parent1subset + parent2subset  


  return child1, child2
  

def generateChildren(matingPool, children):
  matingPool.sort(key = lambda x : x.distance)
  global chromosomeRollNo
  length = len(matingPool) - 1
  # for i in range(1,length, 1):
  #   parent1 = matingPool[0]
  #   parent2 = matingPool[i+1]
  
  for i in range(0,length, 2):
    parent1 = matingPool[i]
    parent2 = matingPool[i+1]
    
    child1, child2 = orderedCrossOver(parent1.route, parent2.route)

    parents = [parent1.chromosomeRollNo, parent2.chromosomeRollNo]
    childChromosome1 = chromosomes(child1,chromosomeRollNo, parents)
    chromosomeRollNo += 1

    parents = [parent2.chromosomeRollNo, parent1.chromosomeRollNo]
    childChromosome2 = chromosomes(child2,chromosomeRollNo, parents)
    chromosomeRollNo += 1

    # if fitnessOperator(childChromosome1) < fitnessOperator(parent1):
    #   children.append(childChromosome1)
    # else:
    #   children.append(parent1)

    # if fitnessOperator(childChromosome2) < fitnessOperator(parent2):
    #   children.append(childChromosome2)
    # else:
    #   children.append(parent2)

    children.append(childChromosome1)
    children.append(childChromosome2)



def mutate(chromosome):
  route = chromosome.route
  routeLength = len(route) 
  position1 = 0
  position2 = 0
  while position1 == position2:
    position1 = random.randint(0,routeLength - 1)
    position2 = random.randint(0,routeLength - 1)
  temp = route[position1]
  route[position1] = route[position2]
  route[position2] = temp
    

def mutateChildren(children, mutatedChildren, mutationRate):
  length = len(children)
  count = int(length * mutationRate)

  while count > 0:
    randomNo = random.randint(0, length - 1)
    pickedChild = children[randomNo]
    if pickedChild not in mutatedChildren:
      mutatedChildren.append(pickedChild)
      count -= 1
  
  for i in mutatedChildren:
    mutate(i)


def createNextGeneration(population, eliteChromosomes, matingPool, children, nextGeneration):
  for i in eliteChromosomes:
    nextGeneration.append(i)
  
  # for i in children:
  #   nextGeneration.append(i)
  
  remainingLength = len(population) - len(nextGeneration)
  population = population + children
  population.sort(key = lambda x : x.distance)
  for i in population:
    if i not in nextGeneration and remainingLength > 0:
      nextGeneration.append(i)
      remainingLength -= 1
  
  
def geneticAlgorithm():
  costList =[]

  initialPopulation = []
  generateInitialPopulation(popSize, initialPopulation, cityList)
  population = initialPopulation.copy()

  for i in range(0,generationNo):
    print("\n")
    print("/////////////////////////////////////////////////////////")
    print("/////_____________ GENERATION: ", i+1 ,"_______________//")
    print("/////////////////////////////////////////////////////////")
    print("\n")
    assignFitness(population)

    print("/////_____________ POPULATION _______________//")
    print(population)

    eliteChromosomes = []
    elitism(population, eliteChromosomes, elitismRate)

    print("/////_____________ ELITE CHROMOSOMES OF THIS POPULATION _______________//")
    print(eliteChromosomes)
    
    matingPool = []
    selectParents(population, matingPool, eliteChromosomes, numberOfParents = popSize * crossOverRate)

    print("/////_____________ SELECTED PARENTS FOR CROSSOVER _______________//")
    print(matingPool)

    children = []
    generateChildren(matingPool, children)
    assignFitness(children)

    print("/////_____________ GENERATED CHILDREN FROM CROSSOVER _______________//")
    print(children)

    mutatedChildren = []
    mutateChildren(children, mutatedChildren, mutationRate)
    assignFitness(mutatedChildren)
    print("/////_____________ APPLYING MUTATION ON CHILDREN _______________//")
    print(mutatedChildren)

    nextGeneration = []
    createNextGeneration(population, eliteChromosomes, matingPool, children, nextGeneration)

    print("/////_____________ NEXT GENERATION = ELITES + CHILDREN + REST OF POPULATION _______________//")
    print(nextGeneration)
    population = nextGeneration.copy()

    sortedPopulation = sorted(population,key = lambda x : x.distance)
    print("/////_____________ BEST CHROMOSOME: ",sortedPopulation[0]," _______________//")
    
    costList.append(sortedPopulation[0].distance)

  return costList



costList = geneticAlgorithm()


#plotting

if plotFlag == 1:
  plt.title("TSP USING GENETIC ALGORITHM\n MIN COST = " + str(costList[-1]))
  plt.xlabel("Generations")
  plt.ylabel("Cost")
  plt.plot(costList, marker="o", mfc="#db513b", mec="#db513b", linestyle = 'dashed', color="#5fcc3d")
  plt.show()









