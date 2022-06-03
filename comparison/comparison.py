import random, time, math
import matplotlib.pyplot as plt
from crossover import *
from mutation import *




# Genetic Algorithm Parameters
popSize = 300
elitismRate = 0.2
mutationRate = .05
crossOverRate = 0.6
generationNo = 100

fileName = "dataSets/ch130.txt"
chromosomeRollNo = 0
plotFlag = 0  # Assign 1 for plotting, 0 for not plotting


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


def readTSPData2 (fileName):
  sourceFile = open(fileName, "r")
  rawData = sourceFile.read()
  sourceFile.close()
  cities = []
  city = []
  temp = ""
  for i in rawData:
    if i != " " and i != "\n":
      temp = temp + i
    
    if i == " " and temp != "":
      city.append(temp)
      temp = ""
    
    if i == "\n" and temp != "":
      city.append(temp)
      temp = ""
      cities.append(city)
      city = []
  
  dataMatrix  = []
  row = []

  for i in range(0, len(cities)):
    row = []
    for j in range(0, len(cities)):
      x1 = float(cities[i][1])
      y1 = float(cities[i][2])
      x2 = float(cities[j][1])
      y2 = float(cities[j][2])
      if i == j:
        temp = 0
      else:
        temp = math.sqrt(  (x1 - x2)**2  +  (y1 - y2)**2   )
      row.append(temp)
    dataMatrix.append(row)
  return dataMatrix


dataMatrix = readTSPData2(fileName)
cityList = [i for i in range(0, len(dataMatrix))]  # City List


class chromosomes:
    def __init__(self, route, chromosomeRollNo, parents=["NA", "NA"]):
        self.chromosomeRollNo = chromosomeRollNo
        self.route = route
        self.distance = 0.0
        self.fitnessScore = 0.0
        self.parents = parents

    def __repr__(self):
        # return " " + str(self.chromosomeRollNo) + ") " + str(self.route) + " Route Distance = "+ str(self.distance) + " Fitness = " + str(self.fitnessScore) + "\n"
        return " " + str(self.chromosomeRollNo) + ") " + str(self.route) + " Cost = " + str(
            self.distance) + " Parents= " + str(self.parents) + "\n"


def generateInitialPopulation(popSize, initialPopulation, cityList):
    global chromosomeRollNo
    count = popSize

    while count > 0:
        chromosome = chromosomes(random.sample(cityList, len(cityList)), chromosomeRollNo)
        if chromosome not in initialPopulation:
            chromosomeRollNo += 1
            initialPopulation.append(chromosome)
            count -= 1


def fitnessOperator(chromosome):
    route = chromosome.route
    totalDistance = 0
    fromCity = 0
    toCity = 0
    for i in range(0, len(route) - 1):
        fromCity = int(route[i])
        toCity = int(route[i + 1])
        totalDistance += dataMatrix[fromCity][toCity]
    fromCity = toCity
    toCity = int(route[0])
    totalDistance += dataMatrix[fromCity][toCity]
    return totalDistance


def assignFitness(population):
    for i in population:
        i.distance = fitnessOperator(i)
        i.fitnessScore = 1 / i.distance


def elitism(population, eliteChromosomes, elitismRate):
    eliteSize = int(len(population) * elitismRate)
    sortedPopulation = sorted(population, key=lambda x: x.fitnessScore, reverse=True)
    for i in range(0, eliteSize):
        eliteChromosomes.append(sortedPopulation[i])


def sus_selection(population, selection_size):

    def selections(population, points):
        keep = []
        
        for p in points:
            i = 0
            while  sum(population[j].fitnessScore for j in range(0, i+1)) < p:
                i += 1
            keep.append(population[i])
        return keep

    total_fitness = 0
    points = []

    for i in population:
        total_fitness += i.fitnessScore

    distance = total_fitness / selection_size

    start_point = random.uniform(0,distance)

    for i in range(0, selection_size):
        points.append(start_point + i*distance)

    return selections(population, points)


def tournament_selection(population):
    participents_no = int(popSize * 30/100)
    selected_participents = []

    while participents_no > 0:
        selected_participents.append(random.choice(population))
        participents_no -= 1
    selected_participents.sort(key=lambda x: x.fitnessScore)
    return selected_participents[-1]


def random_selection(population):
    return random.choice(population)


def rank_selection(population):
    temp = population.copy()
    temp.sort(key=lambda x: x.fitnessScore)
    return temp[-1]


def roulette_wheel_selection(population):
    totalFitness = 0
    for i in population:
        totalFitness += i.fitnessScore
    P = random.random()
    N = 0.0
    for i in population:
        N += i.fitnessScore / totalFitness
        if (N > P):
            return i


def selectParents(population, matingPool, eliteChromosomes, numberOfParents):
    count = numberOfParents
    temp = population.copy()

    while count > 0:

        # selectedParent = rwSelection(temp)
        # selectedParent = tournament_selection(temp)
        # selectedParent = random_selection(temp)
        selectedParent = roulette_wheel_selection(temp)

        if selectedParent not in matingPool:
            matingPool.append(selectedParent)
            count -= 1
            temp.remove(selectedParent)

###########################################################################
###############  THE FOLLOWING PART IS FOR SUS SELECTION  #################
###########################################################################

    # selectedParent = (sus_selection(temp, int(numberOfParents))).copy()
    # i = 0
    # while count > 0:
    #     if selectedParent[i] not in matingPool:
    #         i += 1
    #         count -= 1
    #     else:
    #         selectedParent = (sus_selection(temp, int(numberOfParents))).copy()
    #         count = numberOfParents
    # for i in selectedParent:
    #     matingPool.append(i)


def ordered_crossOver(parent1, parent2):
    randomPoint1 = random.randint(0, (len(parent1) - 1))
    randomPoint2 = random.randint(0, (len(parent1) - 0))

    startGene = min(randomPoint1, randomPoint2)
    endGene = max(randomPoint1, randomPoint2)

    # child 1
    child1 = []
    parent1subset = []
    parent2subset = []

    for i in range(startGene, endGene):
        parent1subset.append(parent1[i])

    parent2subset = [item for item in parent2 if item not in parent1subset]
    child1 = parent1subset + parent2subset

    # child 2
    child2 = []
    parent1subset = []
    parent2subset = []

    for i in range(startGene, endGene):
        parent2subset.append(parent2[i])

    parent1subset = [item for item in parent1 if item not in parent2subset]
    child2 = parent1subset + parent2subset

    return child1, child2


def cyclic_crossover(parent1, parent2):
    length = len(parent1)
    child = [-1] * length
    cycle_index = []
    start_pos = random.randint(0, length - 1)
    temp = start_pos
    next_pos = start_pos
    if start_pos == length - 1:
        temp = start_pos - 1
    elif (start_pos == 0):
        temp = start_pos + 1
    else:
        temp = 0
    while (parent1[temp] != parent1[start_pos]):
        cycle_index.append(next_pos)
        next_pos = parent1.index(parent2[next_pos])
        temp = next_pos
    for i in cycle_index:
        child[i] = parent1[i]
    for i in range(0, length):
        if (i in cycle_index):
            continue
        else:
            child[i] = parent2[i]

    child1 = child.copy()

    # creating the second child
    child = [-1] * length
    for i in cycle_index:
        child[i] = parent2[i]
    for i in range(0, length):
        if (i in cycle_index):
            continue
        else:
            child[i] = parent1[i]

    child2 = child.copy()

    return child1, child2


def order_based_crossover(parent1, parent2):
    length = len(parent1)
    child = [-1] * length
    child_subset = []
    positions = []
    temp_parent = parent1.copy()
    temp_num = random.randint(2, length - 3)
    # print("Number of random positions -", temp_num)
    temp = 0
    for i in range(0, temp_num):
        pos = random.randint(0, length - 1)
        while (temp_parent[pos] == -1):
            pos = random.randint(0, length - 1)
        child_subset.append(temp_parent[pos])
        positions.append(pos)
        temp_parent[pos] = -1
    #   print("random positions are     -", pos)
    # print("Child subset —", child_subset)
    for i in range(0, length):
        if (parent2[i] not in child_subset):
            child[i] = parent2[i]
    for i in range(0, length):
        if child[i] != -1:
            continue
        else:
            child[i] = child_subset[temp]
            temp += 1
    child1 = child.copy()

    # creating the second child--

    child = [-1] * length
    child_subset = []
    temp_parent = parent2.copy()
    temp = 0
    for i in positions:
        child_subset.append(temp_parent[i])
    #   print("random positions are     -", positions)
    # print("Child subset —", child_subset)
    for i in range(0, length):
        if parent1[i] not in child_subset:
            child[i] = parent1[i]
    for i in range(0, length):
        if child[i] != -1:
            continue
        else:
            child[i] = child_subset[temp]
            temp += 1
    child2 = child.copy()

    return child1, child2


def partially_mapped_crossover(parent1, parent2):
    length = len(parent1)
    child = [-1] * length
    subset_len = random.randint(2, length - 2)
    temp = 0

    p1_cutpoint = random.randint(0, length - 1)
    while p1_cutpoint + subset_len > length - 1:
        p1_cutpoint = random.randint(0, length - 1)

    p2_cutpoint = random.randint(0, length - 1)
    while p2_cutpoint + subset_len > length - 1:
        p2_cutpoint = random.randint(0, length - 1)

    temp1, temp2 = p1_cutpoint, p2_cutpoint

    # print("Two cutpoints are -", p1_cutpoint, p2_cutpoint)
    # print("Subset length is —-", subset_len)

    for i in range(0, subset_len):
        child[p1_cutpoint] = parent2[p2_cutpoint]
        p1_cutpoint += 1
        p2_cutpoint += 1

    p1_cutpoint, p2_cutpoint = temp1, temp2

    for i in range(0, length):
        if parent1[i] not in child:
            if child[i] == 0:
                child[i] = parent1[i]

    for i in range(0, length):
        if child[i] == -1:
            while parent1[temp] in child:
                temp += 1
            child[i] = parent1[temp]
    child1 = child.copy()

    # making the second child
    child = [-1] * length
    temp = 0

    for i in range(0, subset_len):
        child[p1_cutpoint] = parent1[p2_cutpoint]
        p1_cutpoint += 1
        p2_cutpoint += 1

    for i in range(0, length):
        if (parent2[i] not in child):
            if child[i] == -1:
                child[i] = parent2[i]

    for i in range(0, length):
        if child[i] == -1:
            while (parent2[temp] in child):
                temp += 1
            child[i] = parent2[temp]
    child2 = child.copy()

    return child1, child2


def position_based_crossover(parent1, parent2):
    length = len(parent1)
    child = [-1] * length
    positions = []
    temp_num = random.randint(2, length - 3)
    # print("Number of random positions -", temp_num)
    temp = 0
    for i in range(0, temp_num):
        pos = random.randint(0, length - 1)
        while (child[pos] != -1):
            pos = random.randint(0, length - 1)
        child[pos] = parent1[pos]
        positions.append(pos)
        # print("Random positions are     -", pos)
    for i in range(0, length):
        while (child[i] == -1):
            if (parent2[temp] not in child):
                child[i] = parent2[temp]
            temp += 1
    child1 = child.copy()

    # creating the second child
    child = [-1] * length
    temp = 0
    for i in positions:
        child[i] = parent2[i]
    # print("Random positions are     -", positions)
    for i in range(0, length):
        while (child[i] == -1):
            if (parent1[temp] not in child):
                child[i] = parent1[temp]
            temp += 1
    child2 = child.copy()

    return child1, child2



def generateChildren(matingPool, children):
    matingPool.sort(key=lambda x: x.distance)
    global chromosomeRollNo
    length = len(matingPool) - 1
    # for i in range(1,length, 1):
    #   parent1 = matingPool[0]
    #   parent2 = matingPool[i+1]

    for i in range(0, length, 2):
        parent1 = matingPool[i]
        parent2 = matingPool[i + 1]

        # child1, child2 = orderedCrossOver(parent1.route, parent2.route)
        # child1, child2 = cyclic_crossover(parent1.route, parent2.route)
        # child1, child2 = order_based_crossover(parent1.route, parent2.route)
        # child1, child2 = partially_mapped_crossover(parent1.route, parent2.route)
        child1, child2 = ordered_crossOver(parent1.route, parent2.route)

        parents = [parent1.chromosomeRollNo, parent2.chromosomeRollNo]
        childChromosome1 = chromosomes(child1, chromosomeRollNo, parents)
        chromosomeRollNo += 1

        parents = [parent2.chromosomeRollNo, parent1.chromosomeRollNo]
        childChromosome2 = chromosomes(child2, chromosomeRollNo, parents)
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


def distinct_checking(populations, index):
    if index > 0:
        temp_index = index - 1
        while temp_index >= 0:
            if populations[index] == populations[temp_index]:
                random.shuffle(populations[index])
                temp_index = index - 1
            else:
                temp_index -= 1


def inversion_mutation(individual):
    def subtour(sequence_list):
        random_subtour = []
        point1 = random.randint(0, len(sequence_list)-1)
        point2 = random.randint(0, len(sequence_list)-1)

        while point1 == point2:
            point2 = random.randint(0, len(sequence_list)-1)

        temp = point1
        point1 = min(point1, point2)
        point2 = max(temp, point2)

        random_subtour = sequence_list[point1 : point2+1]
        random_subtour.reverse()

        return random_subtour, point1, point2

    random_subtour, p1, p2 = subtour(individual.route)
    individual.route[p1 : p2+1] = random_subtour[:]


def insertion_mutation(individual):
    temp = individual.route.pop(random.randint(0, len(individual.route)-1))
    individual.route.insert(random.randint(0, len(individual.route)-1), temp)


def exchange_mutation(individual):
    def point_picker(sequence_list):
        point1 = random.randint(0, len(sequence_list)-1)
        point2 = random.randint(0, len(sequence_list)-1)

        while point1 == point2:
            point2 = random.randint(0, len(sequence_list)-1)

        return point1, point2

    p1, p2 = point_picker(individual.route)
    temp = individual.route[p1]
    individual.route[p1] = individual.route[p2]
    individual.route[p2] = temp
    

def heuristic_mutation(individual):
    three_points = []
    permutes = []
    temp_route = individual.route.copy()

    min_dist = individual.distance

    for i in range(0, 3):
        temp = random.randint(0, len(temp_route) - 1)
        while temp in three_points:
            temp = random.randint(0, len(temp_route) - 1)
        three_points.append(temp)

    random_cities = []
    for index in three_points:
        random_cities.append(temp_route[index])
    permutes.append(random_cities)

    for i in range(1, 6):
        permutes.append(random.sample(random_cities, 3))
        distinct_checking(permutes, i)

    back_up = individual.route.copy()

    for item in permutes:
        temp_route[three_points[0]] = item[0]
        temp_route[three_points[1]] = item[1]
        temp_route[three_points[2]] = item[2]

        individual.route = temp_route
        individual.distance = fitnessOperator(individual)

        if individual.distance < min_dist:
            individual.route = temp_route.copy()
            back_up = temp_route.copy()
        else:
            individual.route = back_up.copy()


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
        exchange_mutation(i)



def createNextGeneration(population, eliteChromosomes, matingPool, children, nextGeneration):
    for i in eliteChromosomes:
        nextGeneration.append(i)

    # for i in children:
    #   nextGeneration.append(i)

    remainingLength = len(population) - len(nextGeneration)
    population = population + children
    population.sort(key=lambda x: x.distance)
    for i in population:
        if i not in nextGeneration and remainingLength > 0:
            nextGeneration.append(i)
            remainingLength -= 1


def geneticAlgorithm():
    costList = []

    initialPopulation = []
    generateInitialPopulation(popSize, initialPopulation, cityList)
    population = initialPopulation.copy()

    for i in range(0, generationNo):
        # print("\n")
        # print("/////////////////////////////////////////////////////////")
        # print("/////_____________ GENERATION: ", i + 1, "_______________//")
        # print("/////////////////////////////////////////////////////////")
        # print("\n")
        assignFitness(population)

        # print("/////_____________ POPULATION _______________//")
        # print(population)

        eliteChromosomes = []
        elitism(population, eliteChromosomes, elitismRate)

        # print("/////_____________ ELITE CHROMOSOMES OF THIS POPULATION _______________//")
        # print(eliteChromosomes)

        matingPool = []
        selectParents(population, matingPool, eliteChromosomes, numberOfParents=popSize * crossOverRate)

        # print("/////_____________ SELECTED PARENTS FOR CROSSOVER _______________//")
        # print(matingPool)

        children = []
        generateChildren(matingPool, children)
        assignFitness(children)

        # print("/////_____________ GENERATED CHILDREN FROM CROSSOVER _______________//")
        # print(children)

        mutatedChildren = []
        mutateChildren(children, mutatedChildren, mutationRate)
        assignFitness(mutatedChildren)
        # print("/////_____________ APPLYING MUTATION ON CHILDREN _______________//")
        # print(mutatedChildren)

        nextGeneration = []
        createNextGeneration(population, eliteChromosomes, matingPool, children, nextGeneration)

        # print("/////_____________ NEXT GENERATION = ELITES + CHILDREN + REST OF POPULATION _______________//")
        # print(nextGeneration)
        population = nextGeneration.copy()

        sortedPopulation = sorted(population, key=lambda x: x.distance)
        # print("/////_____________GEN: ", i+1, " BEST CHROMOSOME: ", sortedPopulation[0], " _______________//")

        costList.append(sortedPopulation[0].distance)

    return costList

takenTime = time.time()
costList = geneticAlgorithm()
takenTime = time.time() - takenTime
print("\nMin cost: ", round(costList[-1], 0), "\tTime Taken: ", round(takenTime, 2))

takenTime = time.time()
costList = geneticAlgorithm()
takenTime = time.time() - takenTime
print("\nMin cost: ", round(costList[-1], 0), "\tTime Taken: ", round(takenTime, 2))

takenTime = time.time()
costList = geneticAlgorithm()
takenTime = time.time() - takenTime
print("\nMin cost: ", round(costList[-1], 0), "\tTime Taken: ", round(takenTime, 2))

takenTime = time.time()
costList = geneticAlgorithm()
takenTime = time.time() - takenTime
print("\nMin cost: ", round(costList[-1], 0), "\tTime Taken: ", round(takenTime, 2))

takenTime = time.time()
costList = geneticAlgorithm()
takenTime = time.time() - takenTime
print("\nMin cost: ", round(costList[-1], 0), "\tTime Taken: ", round(takenTime, 2))

# plotting

if plotFlag == 1:
    plt.title("TSP USING GENETIC ALGORITHM\n MIN COST = " + str(costList[-1]))
    plt.xlabel("Generations")
    plt.ylabel("Cost")
    plt.plot(costList, marker="o", mfc="#db513b", mec="#db513b", linestyle = 'dashed', color="#5fcc3d")
    plt.show()
