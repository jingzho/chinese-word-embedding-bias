import json
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

def load_professionals():
    prof_file = os.path.join(PKG_DIR, '../data', 'professionals.json')
    with open(prof_file, 'r') as f:
        profs = json.load(f)
    return profs

def load_adjectives():
    adjs_file = os.path.join(PKG_DIR, '../data', 'adjectives.json')
    with open(adjs_file, 'r') as f:
        adjs = json.load(f)
    return adjs