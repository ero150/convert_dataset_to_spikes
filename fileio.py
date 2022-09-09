import pickle
import numpy as np
import json

def load_json(dataset_path,inputs_label,labels_label):
  with open(dataset_path, "r") as fp:
    data = json.load(fp)


  # convert lists to numpy arrays
  inputs = np.array(data[inputs_label])
  targets = np.array(data[labels_label])
  del(data)
  return inputs, targets


def load_pickle(dataset_path):
  a_file = open(dataset_path, "rb")
  input_file = pickle.load(a_file)

  inputs,targets = input_file['inputs'],input_file['targets']
  a_file.close()
  del(input_file)

  return inputs,targets


def add_to_text_file(text_file,index,label):
  text_file.write(str(index)+"\t"+(str(label))+"\n")