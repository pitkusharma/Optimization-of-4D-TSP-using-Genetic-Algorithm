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
    return   str(self.x) + "," + str(self.y)  


class routes:
  def __init__(self,path):
    self.path = path
    self.distance = 0.0
    self.fitness = 0.0
  
  def pathDistance(self):
    if self.distance == 0:
      sumDistance = 0
      fromCity = None
      toCity = None

      for i in range(0, len(self.path)+ 1):
        fromCity = self.path[i]
        if i < len(self.path):
          toCity = self.path[i+1]
        else:
          toCity = self.path[0]
        
        sumDistance += fromCity.distance(toCity)
      
      self.distance = float(sumDistance)
    return self.distance
  
  def pathFitness(self):
    if self.fitness == 0:
      self.fitness = 1 / self.distance
    return self.fitness


cityList = []
cityCount = 10
popSize = 10
for i in range(0,cityCount):
    cityList.append(cities(int(random.random() * 100),int(random.random() * 100)))


def createPath(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
  initialPopulation = []
  for _ in range(0,popSize):
    initialPopulation.append(createPath(cityList))
  return initialPopulation

veryFirstPopulation = initialPopulation(popSize, cityList)

print(cityList)
print(veryFirstPopulation)






    
plt.title("TSP USING GENETIC ALGORITHM")
x = []
y = []
for i in range (0, popSize):
  xx = []
  yy = []
  for j in range(0,cityCount):
    objStr = str(veryFirstPopulation[i][j])
    cordinates = objStr.rsplit(",")
    xx.append(int(cordinates[0]))
    yy.append(int(cordinates[1]))
  x.append(xx)
  y.append(yy)
  plt.plot(x[i],y[i], marker="o", mfc="#db513b", mec="#db513b", linestyle = 'dashed', color="#5fcc3d")
plt.show()