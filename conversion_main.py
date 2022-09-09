from encoding import encode_mfcc2 as converter
from fileio import load_pickle
from scale_inputs2 import scale_inputs2 as scaler
import numpy as np

DATASET_PATH = '/media/zalman/b72ea14f-9d1d-432c-9880-6e7f2f40db86/gtzan_spikes/mel_spec/data_mel_spec_test.pkl'
SAVE_PATH = '/media/zalman/b72ea14f-9d1d-432c-9880-6e7f2f40db86/gtzan_spikes/mel_spec/gtzan_rate/'

global_min = -85.36690521240234
global_max = 39.523799896240234
inputs,targets = load_pickle(dataset_path=DATASET_PATH)

inputs_scaled = scaler(inputs=inputs,absolute_max=global_max,absolute_min=global_min,new_max=1,new_min=0)

converter(inputs=inputs_scaled,labels=targets,sample_duration=300,num_steps=100,train_test='test',new_path=SAVE_PATH)

