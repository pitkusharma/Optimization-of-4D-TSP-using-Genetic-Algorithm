import random,copy
from data_functions import *
import matplotlib.pyplot as plt




#Genetic Algorithm Parameters
popSize = 50
elitismRate = 0.2
mutationRate = .05
crossOverRate = 0.8
generationNo = 100




plotFlag = 1 # Assign 1 for plotting, 0 for not plotting
chromosomeRollNo = 0

c_size = 10
r_size = 5
v_size = 3

dataMatrix = generate_data_matrix(c_size,r_size,v_size)
# print_datamatrix(dataMatrix)

cityList = generate_city(c_size)
roadList = generate_road(r_size)
vehicleList = generate_vehicle(v_size)


class chromosomes:
  def __init__(self,route,vehicle,road,chromosomeRollNo, parents=["NA","NA"]):
    self.chromosomeRollNo = chromosomeRollNo
    self.route = route
    self.vehicle = vehicle
    self.road = road
    self.distance = 0.0
    self.fitnessScore = 0.0
    self.parents = parents
    
  def __repr__(self):
    #return " " + str(self.chromosomeRollNo) + ") " + str(self.route) + " Route Distance = "+ str(self.distance) + " Fitness = " + str(self.fitnessScore) + "\n"
    return "\n " + str(self.chromosomeRollNo) + ")  " + str(self.route) + '    ' + str(self.road) + '   ' + str(self.vehicle)  + "        Cost = " + str(self.distance) + "    Parents= "+ str(self.parents) 


def generateInitialPopulation(popSize, initialPopulation, cityList, vehicleList, roadList):
  global chromosomeRollNo
  count = popSize
  
  for i in range(0,count):
    
    temp1 = randomShuffle(cityList)
    temp2 = []
    temp3 = []
    for j in range (0, len(temp1)):
      temp2.append(random.choice(vehicleList))
      temp3.append(random.choice(roadList))

    chromosome = chromosomes(temp1, temp2, temp3 , chromosomeRollNo)
    chromosomeRollNo += 1

    initialPopulation.append(chromosome)


def fitnessOperator(chromosome):
  tempRoute = chromosome.route
  tempVehicle = chromosome.vehicle
  tempRoad = chromosome.road

  totalCost = 0
  
  fromCity = 0
  toCity = 0
  usedVehicle = 0
  usedRoad = 0
  for i in range(0,len(tempRoute)):
    fromCity = int( tempRoute[i] )
    
    if(i+1 == len(tempRoute)):
      toCity = int( tempRoute[0] )
    else:
      toCity = int( tempRoute[i + 1] )
    usedVehicle = tempVehicle[i]
    usedRoad = tempRoad[i]

    totalCost += dataMatrix[fromCity][toCity][usedRoad][usedVehicle]

  return totalCost


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
  bothChild = []
  
  randomPoint1 = random.randint(0,(len(parent1) - 1))
  randomPoint2 = random.randint(0,(len(parent1) - 0))
  
  startGene = min(randomPoint1, randomPoint2)
  endGene = max(randomPoint1, randomPoint2)
  
  #child 1
  child = []
  parent1subset = []
  parent2subset = []
  
  for i in range(startGene, endGene):
    parent1subset.append(parent1[i])
  
  parent2subset = [item for item in parent2 if item not in parent1subset]
  child = parent1subset + parent2subset

  bothChild.append(child)

  #child 2
  child = []
  parent1subset = []
  parent2subset = []

  for i in range(startGene, endGene):
    parent2subset.append(parent2[i])
  
  parent1subset = [item for item in parent1 if item not in parent2subset]
  child = parent1subset + parent2subset  

  bothChild.append(child)

  return bothChild
  

def generateChildren(matingPool, children):
  global chromosomeRollNo
  count = len(matingPool)

  for i in range(0, count, 2):
    parent1 = matingPool[i]
    print("\n\n/////////////////////////////////")
    print("Parent chromosome A: ", parent1.chromosomeRollNo ,")  ", parent1.route, parent1.road, parent1.vehicle)
    parent2 = matingPool[i+1]
    print("Parent chromosome B: ", parent2.chromosomeRollNo ,")  ", parent2.route, parent2.road, parent2.vehicle, "\n")
    childrenStrings = orderedCrossOver(parent1.route,parent2.route)

    parents = [parent1.chromosomeRollNo, parent2.chromosomeRollNo]
    child1 = chromosomes(childrenStrings[0].copy(), parent1.vehicle.copy(), parent1.road.copy(), chromosomeRollNo, parents)
    chromosomeRollNo += 1
    print("\nChildren chromosome A: ", child1.chromosomeRollNo ,")  ", child1.route, child1.road, child1.vehicle)
    parents = [parent2.chromosomeRollNo, parent1.chromosomeRollNo]
    child2 = chromosomes(childrenStrings[1].copy(), parent2.vehicle.copy(), parent2.road.copy(), chromosomeRollNo, parents)
    print("Children chromosome B: ", child2.chromosomeRollNo ,")  ", child2.route, child2.road, child2.vehicle)
    chromosomeRollNo += 1
    #print("/////////////////////////////////")
    children.append(child1)
    children.append(child2)


def swap(list, pos1, pos2):  
  list[pos1], list[pos2] = list[pos2], list[pos1]
  return list


def mutate(chromosome):
  listLength = len(chromosome.route)
  randomPoint1 = random.randint(0, listLength - 1)
  randomPoint2 = random.randint(0, listLength - 1)
  if ( randomPoint1 != randomPoint2):
    print("\n\n/////////////////////////////////")
    print("Before Mutation: ", chromosome.chromosomeRollNo ,")  ", chromosome.route, chromosome.road, chromosome.vehicle)
    print("Random point 1: ", randomPoint1)
    print("Random point 2: ", randomPoint2)
    swap(chromosome.route, randomPoint1, randomPoint2)
    swap(chromosome.road, randomPoint1, randomPoint2)
    swap(chromosome.vehicle, randomPoint1, randomPoint2)
    print("After Mutation: ", chromosome.chromosomeRollNo ,")  ", chromosome.route, chromosome.road, chromosome.vehicle)
  else:
    mutate(chromosome)


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


def createNextGeneration(population, eliteChromosomes, children, nextGeneration):
  for i in eliteChromosomes:
    nextGeneration.append(i)
  
  for i in children:
    nextGeneration.append(i)
  
  remainingLength = len(population) - len(nextGeneration)
  #population = population + children
  population.sort(key = lambda x : x.distance)
  for i in population:
    if i not in nextGeneration and remainingLength > 0:
      nextGeneration.append(i)
      remainingLength -= 1


def geneticAlgorithm():
  costList =[]

  #numberOfParents = popSize * crossOverRate
  initialPopulation = []
  generateInitialPopulation(popSize,initialPopulation,cityList, vehicleList, roadList)
  population = []
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
    createNextGeneration(population, eliteChromosomes, children, nextGeneration)

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







