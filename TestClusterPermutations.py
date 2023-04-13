import itertools
from sklearn.metrics import f1_score

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

ori_number = [0,0,2,2,1,2,0,2]

numbers = [0,0,1,1,2,2,0,1]

unique_values = set(numbers)

permutations = list(itertools.permutations(unique_values))

for perm in permutations:
    replacements = {}
    
    for i in range(len(unique_values)):
        replacements[i] = perm[i]
    new_numbers = replace_numbers(numbers, replacements)
    
    print(new_numbers, f1_score(ori_number, new_numbers, average='weighted'))
