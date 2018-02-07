import io
import multiprocessing as mp
import os
import pickle
from glob import glob
import numpy as np
from datetime import datetime
import struct
import time
import pandas as pd
import sys
import itertools as it


n_cores = 16
res_abs = mp.Manager().list()
res_ent = mp.Manager().list()

with open('/home/mephistopheies/storage2/data/camera-model-id/tmp/p_test_985_m1_u0.pkl', 'rb') as f:
    p_test = pickle.load(f)
classes = sorted([k for (k, _) in list(p_test.values())[0]])
Q = np.zeros((len(p_test), len(classes)), dtype=np.float32)
files = []
for fname, d in p_test.items():    
    d = dict(d)
    for ix, c in enumerate(classes):
        Q[len(files), ix] = d[c]
    files.append(fname)

grid = np.arange(0.6, 1.5, 0.2)
grid = list(it.product(*([grid]*10)))
print(len(grid))


def process(q, iolock):
    n = 0
    
    min_abs = 999
    w_min_abs = None
    
    max_ent = -1
    w_max_ent = None
    
    
    while True:
        w = q.get()
        if w is None:
            res_abs.append((min_abs, w_min_abs))
            res_ent.append((max_ent, w_max_ent))
            break
            
        w = np.array(w)
        P = Q*w
        P = P/P.sum(axis=1)[:, np.newaxis]
        dist = P.sum(axis=0)/264
        z = np.abs(1 - dist).sum()
        e = -(z*np.log(z)).sum()
        if z < min_abs:
            min_abs = z
            w_min_abs = w                
        if e > max_ent:
            max_ent = e
            w_max_ent = w
            
        n += 1
        if n % 1000000 == 0:
            print(n)
            
q = mp.Queue(maxsize=n_cores)
iolock = mp.Lock()
pool = mp.Pool(n_cores, initializer=process, initargs=(q, iolock))

for ix, w in enumerate(grid):
    q.put(w)

for _ in range(n_cores):  
    q.put(None)
pool.close()
pool.join()

res_abs = sorted(list(res_abs), key=lambda t: t[0])[0][1]
res_ent = sorted(list(res_ent), key=lambda t: t[0])[-1][1]
print(res_abs)
print(res_ent)

with open('/home/mephistopheies/storage2/data/camera-model-id/tmp/wbf.pkl', 'wb') as f:
    pickle.dump({
        'abs': res_abs,
        'ent': res_ent,
    }, f)