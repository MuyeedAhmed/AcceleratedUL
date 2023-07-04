import itertools
from sklearn.metrics import f1_score
import numpy as np
# numbers = [0, 1, 2]
# permutations = list(itertools.permutations(numbers))
# print(permutations)


def replace_numbers(numbers, replacements):
    new_numbers = []
    for number in numbers:
        if number in replacements:
            new_numbers.append(replacements[number])
        else:
            new_numbers.append(number)
    return new_numbers

ori_number = [0,0,2,2,1,2,0,3]

numbers = [[0,0,1,1,2,2,0,1],
           [0,0,2,0,1,1,0,1],
           [2,2,1,1,0,2,1,1]]

f1Scores = []
for n_i in range(len(numbers)):
    unique_values = set(numbers[n_i])
    
    permutations = list(itertools.permutations(unique_values))
    
    
    bestPerm = []
    bestF1 = -1
    
    
    for perm in permutations:
        replacements = {}
        
        for i in range(len(unique_values)):
            replacements[i] = perm[i]
        new_numbers = replace_numbers(numbers[n_i], replacements)
        
        f1_s = f1_score(ori_number, new_numbers, average='weighted')
        if f1_s > bestF1:
            bestF1 = f1_s
            bestPerm = new_numbers
            
        print(new_numbers, f1_score(ori_number, new_numbers, average='weighted'))
    print(bestF1)
    f1Scores.append(bestF1)
    numbers[n_i] = bestPerm
    
print(numbers)
a = np.array(numbers)
print(a.mean(axis=0))

for i in range(len(numbers[0])):
    dict={}
    for j in range(len(numbers)):
        if numbers[j][i] in dict:
            dict[numbers[j][i]] += f1Scores[j]
        else:
            dict[numbers[j][i]] = f1Scores[j]
    ori_number[i] = max(dict, key=dict.get)
    print(max(dict, key=dict.get))
    print(dict)
print(ori_number)