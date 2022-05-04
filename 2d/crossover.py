import random


# Implementation of Order Based Crossover (OX 2):

def ordered_crossover(parent1, parent2):
  length = len(parent1)
  child = [0]*length
  child_subset = []
  positions = []
  temp_parent = parent1.copy()
  temp_num = random.randint(2, length-3)
  print("Number of random positions -", temp_num)
  temp = 0
  for i in range(0, temp_num):
    pos = random.randint(0, length-1)
    while (temp_parent[pos] == 0):
      pos = random.randint(0, length-1)
    child_subset.append(temp_parent[pos])
    positions.append(pos)
    temp_parent[pos] = 0
  #   print("random positions are     -", pos)
  # print("Child subset —", child_subset)
  for i in range(0, length):
    if (parent2[i] not in child_subset):
      child[i] = parent2[i]
  for i in range(0, length):
    if child[i] != 0:
      continue
    else:
      child[i] = child_subset[temp]
      temp += 1
  child1 = child.copy()

  # making the second child

  child = [0]*length
  child_subset = []
  temp_parent = parent2.copy()
  temp = 0
  for i in positions:
    child_subset.append(temp_parent[i])
  # print("random positions are     -", positions)
  # print("Child subset —", child_subset)
  for i in range(0, length):
    if (parent1[i] not in child_subset):
      child[i] = parent1[i]
  for i in range(0, length):
    if child[i] != 0:
      continue
    else:
      child[i] = child_subset[temp]
      temp += 1
  child2 = child.copy()

  return child1, child2
  
  


# Implementation of Cyclic Crossover:

def cyclic_crossover(parent1, parent2):
  length = len(parent1)
  child = [0]*length
  cycle_index = []
  start_pos = random.randint(0, length-1)
  temp = start_pos
  next_pos = start_pos
  if (start_pos == length-1):
    temp = start_pos-1
  elif (start_pos == 0):
    temp = start_pos+1
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
  child = [0]*length
  for i in cycle_index:
    child[i] = parent2[i]
  for i in range(0, length):
    if (i in cycle_index):
      continue
    else:
      child[i] = parent1[i]

  child2 = child.copy()

  return child1, child2
  



# Implementation of Order Based Crossover (OX 2):

def order_based_crossover(parent1, parent2):
  length = len(parent1)
  child = [0]*length
  child_subset = []
  positions = []
  temp_parent = parent1.copy()
  temp_num = random.randint(2, length-3)
  # print("Number of random positions -", temp_num)
  temp = 0
  for i in range(0, temp_num):
    pos = random.randint(0, length-1)
    while (temp_parent[pos] == 0):
      pos = random.randint(0, length-1)
    child_subset.append(temp_parent[pos])
    positions.append(pos)
    temp_parent[pos] = 0
  #   print("random positions are     -", pos)
  # print("Child subset —", child_subset)
  for i in range(0, length):
    if (parent2[i] not in child_subset):
      child[i] = parent2[i]
  for i in range(0, length):
    if child[i] != 0:
      continue
    else:
      child[i] = child_subset[temp]
      temp += 1
  child1 = child.copy()

  # creating the second child--

  child = [0]*length
  child_subset = []
  temp_parent = parent2.copy()
  temp = 0
  for i in positions:
    child_subset.append(temp_parent[i])
  #   print("random positions are     -", positions)
  # print("Child subset —", child_subset)
  for i in range(0, length):
    if (parent1[i] not in child_subset):
      child[i] = parent1[i]
  for i in range(0, length):
    if child[i] != 0:
      continue
    else:
      child[i] = child_subset[temp]
      temp += 1
  child2 = child.copy()

  return child1, child2
  
  


# Implementation of Partially Mapped Crossover (PMX):

def partially_mapped_crossover(parent1, parent2):
  length = len(parent1)
  child = [0]*length
  subset_len = random.randint(2, length-2)
  temp = 0
  
  p1_cutpoint = random.randint(0, length-1)
  while (p1_cutpoint + subset_len > length-1):
    p1_cutpoint = random.randint(0, length-1)
  
  p2_cutpoint = random.randint(0, length-1)
  while (p2_cutpoint + subset_len > length-1):
    p2_cutpoint = random.randint(0, length-1)

  temp1, temp2 = p1_cutpoint, p2_cutpoint

  print("Two cutpoints are -", p1_cutpoint, p2_cutpoint)
  print("Subset length is —-", subset_len)
  
  for i in range(0, subset_len):
    child[p1_cutpoint] = parent2[p2_cutpoint]
    p1_cutpoint += 1
    p2_cutpoint += 1

  p1_cutpoint, p2_cutpoint = temp1, temp2

  for i in range(0, length):
    if(parent1[i] not in child):
      if child[i]==0:
        child[i] = parent1[i]

  for i in range(0, length):
    if child[i] == 0:
      while (parent1[temp] in child):
        temp += 1
      child[i] = parent1[temp]
  child1 = child.copy()

# making the second child
  child = [0]*length
  temp = 0

  for i in range(0, subset_len):
    child[p1_cutpoint] = parent1[p2_cutpoint]
    p1_cutpoint += 1
    p2_cutpoint += 1
    
  for i in range(0, length):
    if(parent2[i] not in child):
      if child[i]==0:
        child[i] = parent2[i]

  for i in range(0, length):
    if child[i] == 0:
      while (parent2[temp] in child):
        temp += 1
      child[i] = parent2[temp]
  child2 = child.copy()

  return child1, child2
  




# Implementation of Position Based Crossover:

def position_based_crossover(parent1, parent2):
  length = len(parent1)
  child = [0]*length
  positions = []
  temp_num = random.randint(2, length-3)
  # print("Number of random positions -", temp_num)
  temp = 0
  for i in range(0, temp_num):
    pos = random.randint(0, length-1)
    while (child[pos] != 0):
      pos = random.randint(0, length-1)
    child[pos] = parent1[pos]
    positions.append(pos)
    # print("Random positions are     -", pos)
  for i in range(0, length):
    while(child[i] == 0):
      if (parent2[temp] not in child):
        child[i] = parent2[temp]
      temp += 1
  child1 = child.copy()

# creating the second child
  child = [0]*length
  temp = 0
  for i in positions:
      child[i] = parent2[i]
  # print("Random positions are     -", positions)
  for i in range(0, length):
    while(child[i] == 0):
      if (parent1[temp] not in child):
        child[i] = parent1[temp]
      temp += 1
  child2 = child.copy()


  return child1,  child2
  
  
