from __future__ import print_function, division
from matplotlib import pyplot as plt

import sys
import json
import random
import numpy as np

import debiaswe as dwe
import debiaswe.we as we
from debiaswe.we import WordEmbedding
from debiaswe.data import load_professionals

'''
Part1: Find Bias
'''
# Step 1: Load Data (Word embedding & professionals)
E = WordEmbedding('./data/file_small.txt')
professionals = load_professionals()

# Step 2: Define regional (Northern China people - Southern China people) direction directly
v_region = E.diff('北京人', '上海人')
v_gender = E.diff('男', '女')

# Step 3: Generating analogies of "Northern people: x :: Southern people: y
# a_region = E.best_analogies_dist_thresh(v_region)
# before_map = {}
output = open("output.txt", "a")
# for (a,b,c) in a_region:
#     before_map[a] = b
#     print(a+"-"+b+"-"+str(c), file=output)

# Step 4: Project professionals words onto the regional & gender dimension.
def project(v):
	sp = sorted([(E.v(p).dot(v), p) for p in professionals])
	print(sp[0:20], file=output)
	print(sp[-20:], file=output)

project(v_region)
project(v_gender)


'''
Part2: Debiaswe
'''
sys.path.insert(0, "./debiaswe")
from debiaswe.debias import debias

# Load some region related word lists to debias
with open('./data/region_definitional_pairs.json', "r") as f:
    region_defs = json.load(f)
print("definitional", region_defs, file=output)

with open('./data/region_equalize_pairs.json', "r") as f:
    region_equalize_pairs = json.load(f)
print("equalize pairs", region_equalize_pairs, file=output)

with open('./data/region_specific_seed.json', "r") as f:
    region_specific_words = json.load(f)
print("regional specific", len(region_specific_words), region_specific_words[:10], file=output)

debias(E, region_specific_words, region_defs, region_equalize_pairs)

'''
Part3: Result analysis
'''
sp_region_debiased = sorted([(E.v(p).dot(v_region), p) for p in professionals])
print("===============region_debiased===============", file=output)
print(sp_region_debiased[0:20], file=output)
print(sp_region_debiased[-20:], file=output)


'''
Similarly, apply to gender
'''
with open('./data/gender_definitional_pairs.json', "r") as f:
    gender_defs = json.load(f)
print("definitional", gender_defs, file=output)

with open('./data/gender_equalize_pairs.json', "r") as f:
    gender_equalize_pairs = json.load(f)
print("equalize pairs", gender_equalize_pairs, file=output)

with open('./data/gender_specific_seed.json', "r") as f:
    gender_specific_words = json.load(f)
print("gender specific", len(gender_specific_words), gender_specific_words[:10], file=output)

debias(E, gender_specific_words, gender_defs, gender_equalize_pairs)

sp_region_gender_debiased = sorted([(E.v(p).dot(v_gender), p) for p in professionals])
print("===============gender_debiased===============", file=output)
print(sp_region_gender_debiased[0:20], file=output)
print(sp_region_gender_debiased[-20:], file=output)



# analogies
# a_region_debiased = E.best_analogies_dist_thresh(v_region)

# for (a,b,c) in a_region_debiased:
#     if a in before_map and b != before_map[a]:
#         print("--------------------------------", file=output)
#         print("Before:", a+"-"+before_map[a], file=output)
#         print("After:", a+"-"+b, file=output)

print("Completed")
f.close() 