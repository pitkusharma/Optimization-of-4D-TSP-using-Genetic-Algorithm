import random


#Genetic Algorithm Parameters
popSize = 10
generationNo = 2
mutationRate = .05
crossOverRate = 0.6
eliteSize = 2




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
    return "\n " + str(self.chromosomeRollNo) + ")  " + str(self.route) + '    ' + str(self.vehicle) + '   ' + str(self.road)  + "        Cost = " + str(self.distance) + "    Parents= "+ str(self.parents) 


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


def elitism(population,eliteChromosomes,eliteSize):
  sortedPopulation =  sorted(population, key= lambda x : x.fitnessScore, reverse = True)
  for i in range(0,eliteSize):
    eliteChromosomes.append(sortedPopulation[i])


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
  for i in range(0,numberOfParents):
    selectedParent = rwSelection(population)
    matingPool.append(selectedParent)


def orderedCrossOver(parent1, parent2):
  child = []
  parent1subset = []
  parent2subset = []
  randomPoint1 = random.randint(0,(len(parent1) - 1))
  randomPoint2 = random.randint(0,(len(parent1) - 0))
  startGene = min(randomPoint1, randomPoint2)
  endGene = max(randomPoint1, randomPoint2)

  for i in range(startGene, endGene):
    parent1subset.append(parent1[i])
  
  parent2subset = [item for item in parent2 if item not in parent1subset]
  child = parent1subset + parent2subset
  return child


def singlePointCrossOver(parent1, parent2):
  child = []
  parent1subset = []
  parent2subset = []
  parentLength = len(parent1)
  randomPoint = random.randint(0, parentLength - 1)
  for i in range(0,randomPoint):
    child.append(parent1[i])
  for i in range(randomPoint,parentLength):
    child.append(parent2[i])
  return child


def generateChildren(matingPool, children, eliteSize, crossOverRate):
  global chromosomeRollNo
  count = len(matingPool)
  random.shuffle(matingPool)

  for i in range(0,count - eliteSize):
    parent1 = matingPool[i]
    parent2 = matingPool[i+1]
    child = 0
    childRoute = []
    childVehicle = []
    childRoad = []

    if(random.random() < crossOverRate):
      childRoute = orderedCrossOver( parent1.route, parent2.route )
      childVehicle = singlePointCrossOver( parent1.vehicle, parent2.vehicle )
      childRoad = singlePointCrossOver( parent1.road, parent2.road )

      parents = [ parent1.chromosomeRollNo, parent2.chromosomeRollNo ]
      
      child = chromosomes( childRoute, childVehicle, childRoad, chromosomeRollNo, parents)
      chromosomeRollNo += 1

    else:
      temp = [parent1,parent2]
      child = random.choice(temp)

    children.append(child)


def mutate1(inputList):
  listLength = len(inputList)
  randomPoint1 = random.randint(0, listLength - 1)
  randomPoint2 = random.randint(0, listLength - 1)
  temp = inputList[randomPoint1]
  inputList[randomPoint1] = inputList[randomPoint2]
  inputList[randomPoint2] = temp


def mutate2(inputList, geneList):
  listLength = len(inputList)
  geneListLength = len(geneList)
  randomPoint1 = random.randint(0, listLength - 1)
  randomPoint2 = random.randint(0, geneListLength - 1)
  inputList[randomPoint1] = geneList[randomPoint2]


def mutateChildren(children, mutationRate):
  childrenLength = len(children)
  for i in range(0, childrenLength):
    if(random.random() < mutationRate):
      mutate1(children[i].route)
      mutate2(children[i].vehicle, vehicleList)
      mutate2(children[i].road, roadList)


def createNextGeneration(eliteChromosomes,children,nextGeneration):
  for i in eliteChromosomes:
    nextGeneration.append(i)
  for i in children:
    nextGeneration.append(i)


def geneticAlgorithm(popSize, eliteSize, mutationRate, crossOverRate, generationNo ):
  costList =[]

  initialPopulation = []
  generateInitialPopulation(popSize,initialPopulation,cityList, vehicleList, roadList)
  population = []
  population = initialPopulation
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

    eliteChromosomes = []
    elitism(population,eliteChromosomes,eliteSize)
    print("/////_____________ ELITE CHROMOSOMES OF THIS POPULATION _______________//")
    print(eliteChromosomes)
    print("\n")

    showProbability(population)
    print("\n")

    matingPool = []
    selectParents(population,matingPool,popSize)
    print("/////_____________ SELECTED PARENTS FOR CROSSOVER _______________//")
    print(matingPool)
    print("\n")

    children = []
    generateChildren(matingPool,children,eliteSize,crossOverRate)
    assignFitness(children)
    print("/////_____________ GENERATED CHILDREN FROM CROSSOVER _______________//")
    print(children)
    print("\n")

    mutateChildren(children,mutationRate)
    assignFitness(children)
    print("/////_____________ APPLIED MUTATION ON CHILDREN _______________//")
    print(children)
    print("\n")

    nextGeneration = []
    createNextGeneration(eliteChromosomes,children,nextGeneration)
    print("/////_____________ NEXT GENERATION = ELITE CHROMOSOMES + CHILDREN TOGETHER _______________//")
    print(nextGeneration)
    print("\n")
    population = nextGeneration

    sortedPopulation = sorted(population,key = lambda x : x.distance)
    print("/////_____________ BEST CHROMOSOME SO FAR : ",sortedPopulation[0]," _______________//")
    
    costList.append(sortedPopulation[0].distance)

  return costList


costList = geneticAlgorithm(popSize, eliteSize, mutationRate, crossOverRate, generationNo)