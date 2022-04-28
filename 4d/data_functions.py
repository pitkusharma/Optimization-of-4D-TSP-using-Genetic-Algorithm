import random



def read_tsp_data(file_name):

    source_file = open(file_name, "r")
    raw_data = source_file.read()
    source_file.close()
    formatted_data = []

    temp = ""
    temp_line = []
    for i in raw_data:
        if i != " ":
            temp += str(i)
        if i == " " or i == "\n":
            if temp != "":
                temp = float(temp)

                temp_line.append(temp)
                temp = ""
        if i == "\n":
            formatted_data.append(temp_line)
            temp_line = []
    temp = float(temp)

    temp_line.append(temp)
    formatted_data.append(temp_line)

    return formatted_data

def generate_city(size):
    temp_city = []
    for i in range(0, size):
        temp_city.append(i)
    return temp_city

def generate_road(size):
    temp_road = []
    for i in range(0, size):
        temp_road.append(i)
    return temp_road

def generate_vehicle(size):
    temp_vehicle = []
    for i in range(0, size):
        temp_vehicle.append(i)
    return temp_vehicle

def print_datamatrix(data_matrix):
    print('\n'+" DATAMATRIX "
         +'\n'+"============")
    print('[')
    for i in range(0, len(data_matrix)):
        print('\t', '[', '\n')
        for j in range(0, len(data_matrix)):
            print('\t', data_matrix[i][j], ',\n')
        print('\t', '],')
    print(']')

def vehicle_assign(v_size):
    vehicle_list = []
    for l in range(0, v_size):
        vehicle_list.append(random.randint(1, 99))
    return vehicle_list
    
def road_assign(r_size, v_size, i, j):
    road_list = []
    if (i == j):
        for k in range(0, r_size):
            road_list.append([0] * v_size)
        return road_list
    for k in range(0, r_size):
        road_list.append(vehicle_assign(v_size))
    return road_list
    
def city_assign(c_size, r_size, v_size, i):
    city_list = []
    for j in range(0, c_size):
        city_list.append(road_assign(r_size, v_size, i, j))
    return city_list
    
def generate_data_matrix(c_size, r_size, v_size):
    data_list = []
    for i in range(0, c_size):
        data_list.append(city_assign(c_size, r_size, v_size, i))
    return data_list
    
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

def distinct_checking(populations, index):
    if index > 0:
        temp_index = index - 1
        while temp_index >= 0:
            if populations[index] == populations[temp_index]:
                random.shuffle(populations[index])
                temp_index = index - 1
            else:
                temp_index -= 1

def factorial(n):
        if n == 1:
            return n
        else:
            return n * factorial(n - 1)

def matrix_2d_to_4d(dataMatrix):
    length = len(dataMatrix)
    temp1 = []
    for i in range(0, length):
        temp2 = []
        for j in range(0, length):
            for k in range(1):
                temp3 = []
                temp2.append(temp3)
                for l in range(1):
                    temp4 = []
                    temp4.append(dataMatrix[i][j])
                    temp3.append(temp4)
        temp1.append(temp2)

    return temp1, length, 1, 1