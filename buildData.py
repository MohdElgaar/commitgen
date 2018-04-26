#!/usr/bin/env python
# -*-coding: utf8 -*-

import numpy as np
from os import path
import json
import pickle
import os
import argparse
from commitgen.data import build_data, split_list, build_vocab

desc = "Help for buildData"
work_dir = os.environ['WORK_DIR']

parser = argparse.ArgumentParser(description=desc)

parser.add_argument("dataset",
                    help="Name or comma-separated names of the pickle dataset file/s (without .pickle) in " + work_dir)

parser.add_argument('--language', "-l",
                    help="Language")

parser.add_argument('--code_max_length', "-cml",
                    type=int,
                    default=100,
                    help="maximum code length")

parser.add_argument('--nl_max_length', "-nml",
                    type=int,
                    default=100,
                    help="maximum nl length")

parser.add_argument('--code_unk_threshold', "-cut",
                    type=int,
                    default=2,
                    help="code unk threshold")

parser.add_argument('--nl_unk_threshold', "-nut",
                    type=int,
                    default=2,
                    help="nl unk threshold")

parser.add_argument("--test", "-t", 
                    action='store_false', 
                    help="To generate a test set (otherwise valid=test)")

parser.add_argument("--ratio", "-r",
                    type=float,
                    default=0.8,
                    help="Train/Test split ratio")

args = parser.parse_args()

work_dir = os.path.join(work_dir, "preprocessing")
if not os.path.isdir(work_dir):
    os.mkdir(work_dir)

if "," in args.dataset:
    datasets = args.dataset.split(",")
else:
    datasets = [args.dataset]
    
if "," in args.language:
    languages = args.language.split(",")
else:
    languages = [args.language]


def process_data(data, test = False):
    if test:
        data = build_data(data, vocab)
        X = []
        Y = []
        Ylen = []
        Xlen = []
        ids = []
        for sample in data:
            x = sample['code_num']
            y = sample['nl_num']
            ids.append(sample['id'])
            X.append(x)
            Y.append(y)
            Xlen.append(sample['code_sizes'])
            Ylen.append(len(sample['nl_num']))
        X = np.array(X)
        Y = np.array(Y)
    else:
        data = build_data(data, vocab,
                          max_code_length=args.code_max_length,
                          max_nl_length=args.nl_max_length)
        X = np.array([])
        ids = []
        for sample in data:
            x = np.ones(args.code_max_length)
            np.put(x, range(sample['code_sizes']), sample['code_num'])
            nl_num = sample['nl_num']
            y = np.ones(args.nl_max_length)
            np.put(y, range(len(nl_num)), nl_num)
            ids.append(sample['id'])
            if X.size == 0:
                X = x
                Y = y
                Xlen = [sample['code_sizes']]
                Ylen = [len(sample['nl_num'])]
                continue
            X = np.vstack((X, x))
            Y = np.vstack((Y, y))
            Xlen.append(sample['code_sizes'])
            Ylen.append(len(sample['nl_num']))
    return X, Y, np.array(Xlen), np.array(Ylen), np.array(ids)

per_dataset_parsed_commits = []
all_parsed_commits = []

for dataset in datasets:
  filepath = os.path.join(work_dir, dataset + ".pickle")
  if os.path.isfile(filepath):
      with open(filepath, "rb") as f:
          parsed_commits = pickle.load(f)
      per_dataset_parsed_commits.append(parsed_commits)
      all_parsed_commits += parsed_commits
  else:
    raise IOError("Pickle file does not exist:" + dataset)

vocab = build_vocab(all_parsed_commits, args.code_unk_threshold, args.nl_unk_threshold)

dataset_name = "_".join(datasets)
language_name =  "_".join(languages)

# storing vocab
vocab_file_name = ".".join([dataset_name, language_name, 'vocab.json'])
with open(path.join(work_dir, vocab_file_name), 'w') as f:
    json.dump(vocab, f)

per_dataset_train = []
per_dataset_valid = []
per_dataset_test = []
all_test = []

for parsed_commits in per_dataset_parsed_commits:
    # splitting dataset
    train, valid, test = split_list(parsed_commits, generate_test=args.test, ratio=args.ratio)
    per_dataset_train.append(train)
    per_dataset_valid.append(valid)
    per_dataset_test.append(test)
    all_test += test

# generating data and saving files

data_to_save = dict()
data_to_save['X'] = dict()
data_to_save['Y'] = dict()
data_to_save['Xlen'] = dict()
data_to_save['Ylen'] = dict()
data_to_save['ids'] = dict()
for i in range(len(datasets)):
    dataset_X, dataset_Y, dataset_Xlen, dataset_Ylen, dataset_ids = process_data(per_dataset_train[i])
    data_to_save['X'][datasets[i]] = dataset_X
    data_to_save['Y'][datasets[i]] = dataset_Y
    data_to_save['Xlen'][datasets[i]] = dataset_Xlen
    data_to_save['Ylen'][datasets[i]] = dataset_Ylen

train_name = ".".join([dataset_name, language_name,"train"])
np.savez(os.path.join(work_dir, train_name), 
         **data_to_save)
print("Successfully generated train data")

data_to_save = dict()
data_to_save['X'] = dict()
data_to_save['Y'] = dict()
data_to_save['Xlen'] = dict()
data_to_save['Ylen'] = dict()
data_to_save['ids'] = dict()
for i in range(len(datasets)):
    dataset_X, dataset_Y, dataset_Xlen, dataset_Ylen, dataset_ids = process_data(per_dataset_valid[i])
    data_to_save['X'][datasets[i]] = dataset_X
    data_to_save['Y'][datasets[i]] = dataset_Y
    data_to_save['Xlen'][datasets[i]] = dataset_Xlen
    data_to_save['Ylen'][datasets[i]] = dataset_Ylen

valid_name = ".".join([dataset_name, language_name, "valid"])
np.savez(os.path.join(work_dir, valid_name), 
         **data_to_save)
print("Successfully generated valid data")



# we don't set a maximum length ONLY for test data
_, ref_data = build_data(all_test, vocab, ref=True)
ref_name = ".".join([dataset_name, language_name, "ref.txt"])

with open(os.path.join(work_dir, ref_name), 'w') as f:
    for i, (sha, nl) in enumerate(ref_data):
        all_test[i] = all_test[i]._replace(id=i)
        try:
            f.write(str(i)+ "\t" + nl + "\n")
        except:
            f.write(str(i)+ "\t" + nl.decode('utf-8').encode('ascii', 'ignore') + "\n")

test_count = 0
for i in range(len(datasets)):
    current_length = len(per_dataset_test[i])
    per_dataset_test[i] = all_test[test_count:test_count+current_length]
    test_count += current_length


data_to_save = dict()
data_to_save['X'] = dict()
data_to_save['Y'] = dict()
data_to_save['Xlen'] = dict()
data_to_save['Ylen'] = dict()
data_to_save['ids'] = dict()
for i in range(len(datasets)):
    dataset_X, dataset_Y, dataset_Xlen, dataset_Ylen, dataset_ids = process_data(per_dataset_test[i], test)
    data_to_save['X'][datasets[i]] = dataset_X
    data_to_save['Y'][datasets[i]] = dataset_Y
    data_to_save['Xlen'][datasets[i]] = dataset_Xlen
    data_to_save['Ylen'][datasets[i]] = dataset_Ylen
    data_to_save['ids'][datasets[i]] = dataset_ids

test_name = ".".join([dataset_name, language_name, "test"])
np.savez(os.path.join(work_dir, test_name), 
         **data_to_save)
print("Successfully generated test data")
