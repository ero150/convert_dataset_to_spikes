import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
import json
import os
from sklearn.model_selection import train_test_split
import pickle
import h5py
import snntorch as snn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import utils
from torch.utils.data import DataLoader
from snntorch import spikegen
from tqdm import tqdm
from time import sleep

def scale_inputs2(inputs, absolute_max, absolute_min, new_max=1, new_min=0):
    # absolute_max = np.max(np.max(inputs))
    # absolute_min = np.min(np.min(inputs))

    scaled = np.array(inputs)
    # print(scaled.shape)
    # print("LEN SCALED",len(scaled))
    # print("LEN SCALED[0]",len(scaled[0]))
    # print("LEN SCALED[0][0]",len(scaled[0][0]))
    # print(absolute_max)
    # print(absolute_min)

    for index_i, i in tqdm(enumerate(np.array(inputs)), total=len(inputs)):
        # sleep(3)
        # print("I length: ",len(i))

        # print(len(i))
        # print("I length", len(i))
        # if index_i ==1 :continue
        # if index_i ==2 :continue
        # if index_i == 2: continue
        # if index_i == 3: continue
        # if index_i == 4: continue
        # if index_i == 5: continue
        # if index_i == 6: continue
        # if index_i == 7: continue
        # if index_i == 8: continue
        # if index_i == 9: continue
        # if index_i == 10: continue
        # if index_i == 3: continue

        # print("Index i: ", index_i)
        for index_j, j in enumerate(np.array(i)):

            # print("Index j: ",index_j,j.shape)
            # print("J lenght",len(j))
            # print("Index ", index_j)
            # print("J:", j)
            iters = 0
            # index_k = 0
            # for k in j:
            #    scaled[index_k][index_j] = (k - absolute_min) / (absolute_max - absolute_min)
            #    index_k += 1
            for index_k, k in enumerate(np.array(j)):
                # print("k lenght", len(k))
                # print(k)
                # print("K: ",index_k)
                # scaled[index_i][index_j][index_k]= (k - absolute_min) / (absolute_max - absolute_min)
                X_std = (k - absolute_min) / (absolute_max - absolute_min)
                scaled[index_i][index_j][index_k] = X_std * (new_max - new_min) + new_min

                # global_min = -66.12835693359375
                # global_max = 37.706790924072266
                # print(scaled[index_i][index_j][index_k])
                # print("")
            # iters+=1
            # print(iters)

    # print("J:",len(j))
    # print(scaled[0][1][0])
    # print(len(scaled), type(scaled), type(scaled[0]), type(scaled[0][0]), type(scaled[0][0][0]))

    return np.array(scaled)
