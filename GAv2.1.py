import random


#Genetic Algorithm Parameters
popSize = 10
generationNo = 2
mutationRate = .3
crossOverRate = 0.6




chromosomeRollNo = 0

# minimum cost till now 59
dataMatrix = [
    [
        [ [00,00,00], [00,00,00], [00,00,00] ],

        [ [43,45,20], [21,35,34], [50,39,42] ],

        [ [40,35,28], [23,53,31], [22,47,31] ]
    ],
    
    [
        [ [43,45,20], [21,35,34], [50,39,42] ],

        [ [00,00,00], [00,00,00], [00,00,00] ],

        [ [31,19,29], [31,21,21], [27,30,17] ]
    ],

    [
        [ [40,35,28], [23,53,31], [22,47,31] ],

        [ [31,19,29], [31,21,21], [27,30,17] ],

        [ [00,00,00], [00,00,00], [00,00,00] ]
    ]    
]


cityList = [0,1,2]
vehicleList = [0,1,2]
roadList = [0,1,2]


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


def randomShuffle(inputList):
  copyList = inputList.copy()
  listLength = len(copyList) 
  
  for i in range (0, listLength):
    point1  = random.randint(0, listLength - 1)
    point2 = random.randint(0, listLength - 1)

    temp = copyList[point1]
    copyList[point1] = copyList[point2]
    copyList[point2] = temp

  return(copyList)


def generateInitialPopulation(popSize, initialPopulation, cityList, vehicleList, roadList):
  global chromosomeRollNo
  count = popSize
  
  for i in range(0,count):
    
    temp1 = randomShuffle(cityList)
    temp2 = randomShuffle(vehicleList)
    temp3 = randomShuffle(roadList)

    chromosome = chromosomes(temp1, temp2, temp3 , chromosomeRollNo)
    chromosomeRollNo += 1

    initialPopulation.append(chromosome)

#initialPopulation = []
#generateInitialPopulation(popSize, initialPopulation, cityList, vehicleList, roadList)
#population = initialPopulation

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

    totalCost += dataMatrix[fromCity][toCity][usedVehicle][usedRoad]

  return totalCost


def assignFitness(population):
  for i in population:
    i.distance = fitnessOperator(i)
    i.fitnessScore = 1/i.distance

#assignFitness(population)
#print(population)


# def elitism(population,eliteChromosomes,eliteSize):
#   sortedPopulation =  sorted(population, key= lambda x : x.fitnessScore, reverse = True)
#   for i in range(0,eliteSize):
#     eliteChromosomes.append(sortedPopulation[i])


def showProbability (population):
  totalFitness = 0
  for i in population:
    totalFitness += i.fitnessScore
  cumulativeProbability = 0
  print("/////_____________ SELECTION PROBABILITY OF POPULATION & CUMULATIVE PROBABILITY _______________//")
  print("\n")
  print("id        Fitness           Selection Probability       Cumulative Probability ")
  print("\n")
  for i in population:
    i.selectProbability = i.fitnessScore / totalFitness
    cumulativeProbability += i.selectProbability
    print( str(i.chromosomeRollNo) + ")     " +  str(i.fitnessScore) + "    " + str(i.selectProbability) + "         " + str(cumulativeProbability))

#showProbability(population)

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


def selectParents(population,matingPool,numberOfParents):
  # for i in range(0,numberOfParents):
  #   selectedParent = rwSelection(population)
  #   matingPool.append(selectedParent)
  if (numberOfParents > 0):
    selectedParent = rwSelection(population)
    if (selectedParent not in matingPool):
      matingPool.append(selectedParent)
      numberOfParents -= 1
      selectParents(population, matingPool, numberOfParents)
    else:
      selectParents(population, matingPool, numberOfParents)

#matingPool = []

#selectParents(population,matingPool,4)
#print(matingPool)


def orderedCrossOver(parent1, parent2):
  children = []
  
  offspring1P1 = []
  offspring1P2 = []
  
  offspring2P1 = []
  offspring2P2 = []
  
  randomPoint1 = random.randint(0,(len(parent1) - 1))
  randomPoint2 = random.randint(0,(len(parent1) - 1))
  
  if(randomPoint1 != randomPoint2):
    print("Parent A city string: ", parent1)
    print("Parent B city string: ", parent2)

    startGene = min(randomPoint1, randomPoint2)
    print("Random Point 1: ", startGene)
    endGene = max(randomPoint1, randomPoint2)
    print("Random Point 2: ", endGene)

    for i in range(startGene, endGene):
      offspring1P1.append(parent1[i])
    print("Sliced part from Parent 1: ", offspring1P1)
    offspring1P2 = [item for item in parent2 if item not in offspring1P1]
    print("Rest elements from Parent 2: ", offspring1P2)
    child1 = offspring1P1 + offspring1P2
    print("Child 1: ", child1)
    

    for i in range(startGene, endGene):
      offspring2P1.append(parent2[i])
  
    print("Sliced part from Parent 2: ", offspring2P1)
    offspring2P2 = [item for item in parent1 if item not in offspring2P1]
    print("Rest elements from Parent 1: ", offspring2P2)
    child2 = offspring2P1 + offspring2P2
    print("Child 2: ", child2)
    
    children.append(child1)
    children.append(child2)

    return children

  else:
    return orderedCrossOver(parent1, parent2)

#print(orderedCrossOver(matingPool[0].route,matingPool[1].route))  

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
    child1 = chromosomes(childrenStrings[0], parent1.vehicle, parent1.road, chromosomeRollNo, parents)
    chromosomeRollNo += 1
    print("\nChildren chromosome A: ", child1.chromosomeRollNo ,")  ", child1.route, child1.road, child1.vehicle)
    parents = [parent2.chromosomeRollNo, parent1.chromosomeRollNo]
    child2 = chromosomes(childrenStrings[1], parent2.vehicle, parent2.road, chromosomeRollNo, parents)
    print("Children chromosome B: ", child2.chromosomeRollNo ,")  ", child2.route, child2.road, child2.vehicle)
    chromosomeRollNo += 1
    #print("/////////////////////////////////")
    children.append(child1)
    children.append(child2)

#children = []
#generateChildren(matingPool, children) 


#print(children)


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

def mutateChildren(children, mutationRate):
  childrenLength = len(children)
  for i in range(0, childrenLength):
    if(random.random() < mutationRate):
      mutate(children[i])

def createNextGeneration(population, matingPool, children, nextGeneration):
  for i in population:
    if i not in matingPool:
      nextGeneration.append(i)
  for i in children:
    nextGeneration.append(i)

#nextGeneration = []

#createNextGeneration(population, matingPool, children, nextGeneration)

#print(nextGeneration)



def geneticAlgorithm(popSize, mutationRate, crossOverRate, generationNo ):
  costList =[]

  numberOfParents = popSize * crossOverRate
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
    print("\n")

    showProbability(population)
    print("\n")

    matingPool = []
    selectParents(population,matingPool,numberOfParents)
    print("/////_____________ SELECTED PARENTS FOR CROSSOVER _______________//")
    print(matingPool)
    print("\n")

    print("/////_____________ CROSSOVER STARTED _______________//")
    children = []
    generateChildren(matingPool,children)
    assignFitness(children)
    print("\n\n/////_____________ GENERATED CHILDREN FROM CROSSOVER _______________//")
    print(children)
    print("\n")
    print("/////_____________ MUTATION STARTED _______________//")
    mutateChildren(children,mutationRate)
    assignFitness(children)
    print("\n\n/////_____________ APPLIED MUTATION ON CHILDREN _______________//")
    print(children)
    print("\n")

    nextGeneration = []
    createNextGeneration(population, matingPool, children, nextGeneration)
    print("/////_____________ NEXT GENERATION _______________//")
    print(nextGeneration)
    print("\n")
    population = nextGeneration.copy()

    sortedPopulation = sorted(population,key = lambda x : x.distance)
    print("/////_____________ BEST CHROMOSOME SO FAR : ",sortedPopulation[0]," _______________//")
    
    costList.append(sortedPopulation[0].distance)

  return costList


costList = geneticAlgorithm(popSize, mutationRate, crossOverRate, 2)