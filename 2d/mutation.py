import random


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
        temp = random.randint(0, len(temp_route)-1)
        while temp in three_points:
            temp = random.randint(0, len(temp_route)-1)
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
