import matplotlib.pyplot as plt, random, numpy as np


class cities:
  def __init__(self,x,y):
    self.x = x
    self.y = y
  
  def distance(self,toCity):
    x = abs(self.x - toCity.x)
    y = abs(self.y - toCity.y)
    distance = np.sqrt(x**2 + y**2)
    return distance
  
  def __repr__(self):
    return   "[" + str(self.x) + "," + str(self.y) + "]" 


class routes:
  def __init__(self,path):
    self.path = path
    self.distance = 0.0
  
  def pathDistance(self):
    sumDistance = 0
    fromCity = None
    toCity = None

    for i in range(0, len(self.path)):
      fromCity = self.path[i]
      if i + 1 < len(self.path):
        toCity = self.path[i+1]
      else:
        toCity = self.path[0]
      
      sumDistance += fromCity.distance(toCity)
    
    self.distance = float(sumDistance)
    return self.distance

  def __repr__(self):
    return str(int(self.pathDistance()))
  

def initialPopulation(popSize, cityList):
  initialPopulation = []
  for _ in range(0,popSize):
    route = routes(random.sample(cityList, len(cityList)))
    initialPopulation.append(route)
  return initialPopulation


def selection(population, eliteSize):
  selectedPopulation = []

  for i in range(eliteSize):
    selectedPopulation.append(population[i])
  count = len(population) - eliteSize
  
  for i in range(count,0,-1):
    index = random.randint(0,i)
    selectedPopulation.append(population[index])

  return selectedPopulation


def crossOver(parent1, parent2):
  childPath = []
  parent1subset = []
  parent2subset = []
  
  randomPoint1 = random.randint(0,(len(parent1.path) - 1))
  randomPoint2 = random.randint(0,(len(parent1.path) - 1))
  
  startGene = min(randomPoint1, randomPoint2)
  endGene = max(randomPoint1, randomPoint2)

  for i in range(startGene, endGene):
      parent1subset.append(parent1.path[i])
      
  parent2subset = [item for item in parent2.path if item not in parent1subset]

  childPath = parent1subset + parent2subset
  child = routes(childPath)
  return child


def generateChildren(matingPool,eliteSize):
  children = []
  count = len(matingPool)
  pool = random.sample(matingPool,count)

  for i in range(0,eliteSize):
    children.append(matingPool[i])
  
  for i in range(0,count - eliteSize):
    parent1 = pool[i]
    parent2 = pool[i+1]
    children.append(crossOver(parent1,parent2))
  return children


def mutate(individual,mutationRate):
  if random.random() < mutationRate:
    path = individual.path
    pathLength = len(path) - 1
    position1 = random.randint(0,pathLength)
    position2 = random.randint(0,pathLength)
    temp = path[position1]
    path[position1] = path[position2]
    path[position2] = temp
    individual.path = path
  return individual


def mutatePopulation(population,mutationRate):
  for i in population:
    i = mutate(i,mutationRate)
  return population


def geneticAlgorithm(population,eliteSize,mutationRate,generationNo):
  costList = []
  for i in range(0,generationNo):
    population.sort(key= lambda x : x.pathDistance())
    minCost = population[0].pathDistance()
    costList.append(minCost)
    print("Generation: "+str(i+1)+" Minimum Cost = "+str(minCost))
    matingPool = selection(population,eliteSize)
    children = generateChildren(matingPool,eliteSize)
    population = mutatePopulation(children,mutationRate)

  print("\n"+"First Gen minimum cost:" + str(costList[0]) + "\n" + "Final Gen minimum cost: " + str(costList[generationNo-1]))
  
  returnList = []
  returnList.append(costList)
  returnList.append(population[0].path)
  return returnList



popSize = 100
cityList = [] #City object List
cityCount = 25
eliteSize = 20
mutationRate = 0.05
generationNo = 200


for i in range(0,cityCount):
  cityList.append(cities(int(random.random() * 100),int(random.random() * 100)))


# cityCordinates = [[0,68], [53,31], [79,26], [67,55], [6,60], [1,38], [66,66], [49,27], [81,24], [19,54], [15,92], [83,47], [31,86], [68,80], [75,85], [79,97], [29,53], [11,47], [26,22], [71,47], [80,63], [11,44], [76,28], [71,98], [83,72]]
# for i in range(0,cityCount):
#   cityList.append(cities(cityCordinates[i][0],cityCordinates[i][1]))


veryFirstPopulation = initialPopulation(popSize, cityList) #Creating Initial population
population = veryFirstPopulation


returnList = geneticAlgorithm(population,eliteSize,mutationRate,generationNo)


#plotting 


plt.title("TSP USING GENETIC ALGORITHM")
plt.xlabel("Generations")
plt.ylabel("Cost")
plt.plot(returnList[0], marker="o", mfc="#db513b", mec="#db513b", linestyle = 'dashed', color="#5fcc3d")
plt.show()


print("\nCITIES CORDINATE LIST: ",cityList)


x = []
y = []
for i in range(0,len(returnList[1])):
  x.append(returnList[1][i].x)
  y.append(returnList[1][i].y)

plt.title("TSP USING GENETIC ALGORITHM\n" + "ROUTE COST = " + str(round(returnList[0][-1], 2)))
plt.plot(x,y, marker="o", mfc="#db513b", mec="#db513b", linestyle = 'dashed', color="#5fcc3d")
plt.show()