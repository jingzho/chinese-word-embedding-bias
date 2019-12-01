'''
Part1: Find Bias
'''
from __future__ import print_function, division

from matplotlib import pyplot as plt
import json
import random
import numpy as np

import debiaswe as dwe
import debiaswe.we as we
from debiaswe.we import WordEmbedding
from debiaswe.data import load_professions

# Step 1: Load Data (Word embedding)
E = WordEmbedding('./data/file_small.txt')

# Step 2.1: Define regional (Northern China people - Southern China people) direction directly
v_region = E.diff('北方人', '南方人')

# Step 3: Generating analogies of "Northern people: x :: Southern people: y
a_region = E.best_analogies_dist_thresh(v_region)
for (a,b,c) in a_region:
    print(a+"-"+b+":"+str(c))

# Step 4: Project character adjectives onto the regional dimension.
adjs = ["漂亮", "完美", "快乐", "聪明", "豁达", "舒服", "自豪", "善良", "自信", "温柔", "靠谱",
       "开心", "幸福", "有趣", "可爱", "不错", "美丽", "有名", "幽默", "友善", "勤奋", "直爽",
       "贤惠", "正直", "坦诚", "活泼", "高尚", "苗条", "纯洁", "得体", "优雅", "敏捷", "热情似火",
       "文质彬彬", "兴致勃勃", "勇往直前", "人高马大", "博学多才", "学富五车", "厚德载物", "积极进取", "拾金不昧", "大公无私",
       "平静", "冷静", "有钱", "贫穷", "仁慈", "温雅", "外向", "内向", "和平", "搞笑", "精明", "高大", "矮小",
       "爱心", "害羞", "努力", "尊重", "正常", "真实", "文明", "忙碌", "悠闲", "时尚", "老土", 
       "怀旧", "漂泊", "感性", "理性", "普通", "独立", "天真", "谨慎", "随和", "淳朴", "政治正确",
       "触景生情", "察言观色", "雄心勃勃", "不吵不闹", "劫富济贫", "谈笑风生", "高谈阔论", "中规中矩", "成熟稳重", "大名鼎鼎",
       "难过", "马虎", "愚蠢", "冷漠", "慌张", "无聊", "恶心", "狡猾", "抑郁", "悲伤", "装逼",
       "不好", "奇怪", "坏", "麻木", "难听", "色情", "可怜", "邪恶", "委屈", "霸道", "小气",
       "花心", "恐怖", "反感", "病态", "虚伪", "自闭", "啰嗦", "自私", "蛮横", "懒惰", "偷鸡摸狗",
       "不思进取", "无所事事", "阴阳怪气", "一塌糊涂", "好吃懒做", "无事生非", "落井下石", "同流合污", "优柔寡断", "心术不正"]

sp = sorted([(E.v(adj).dot(v_region), adj) for adj in adjs])
print(sp[0:20], sp[-20:])


'''
Part2: Debias
'''
import sys
sys.path.insert(0, "./debiaswe")
from debiaswe.debias import debias

# Load some region related word lists to debias
with open('./data/definitional_pairs.json', "r") as f:
    defs = json.load(f)
print("definitional", defs)

with open('./data/equalize_pairs.json', "r") as f:
    equalize_pairs = json.load(f)

with open('./data/gender_specific_seed.json', "r") as f:
    gender_specific_words = json.load(f)
print("gender specific", len(gender_specific_words), gender_specific_words[:10])

debias(E, gender_specific_words, defs, equalize_pairs)
