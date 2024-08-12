#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:31:41 2024

@author: sarabcidf
"""

# %% Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import resource
import os
import pickle

from tqdm import tqdm
from collections import defaultdict

from nltk.tokenize import sent_tokenize
from IPython.display import clear_output

import sys
import time
import os
import csv
import itertools
import json
import scipy


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error

# pip install nltk
# pip install tqdm
# nltk.download('punkt')

# %% WD

working_directory = '/home/users/cids'
os.chdir(working_directory)
print("Current working directory:", os.getcwd())

# %% Reading data

data = pd.read_csv('sentences.csv')

# %% Exploring and cleaning

data
data.head()
data.columns 
print(data.iloc[0])

# Getting rid of NAs in target: 

print("NaNs:", data['clean_text'].isna().sum())
data['clean_text'] = data['clean_text'].fillna('')

# %% Filtering and sampling for Mexico

data_mex = data[data['gps_ISO'] == 'MEX']
data_mex = data_mex.sample(n=5000, random_state=42)

# %% Creating BoW (Modified for Regression)

# Calculating the total number of sentences in the dataset
N_sentences = len(data_mex)

# Initializing a dictionary to count occurrences of each word
counts = defaultdict(int)

# Iterating through each sentence to count word frequencies
for index, row in data_mex.iterrows():
    # Splitting the cleaned text into words
    clean_text = row['clean_text'].split()
    # Counting each word's occurrence
    for word in clean_text:
        counts[word] += 1

# Filtering out infrequent and short words
to_del = [word for word in counts if counts[word] <= 4 or len(word) <= 2]
for word in to_del:
    del counts[word]

# Creating a list of words that remain after filtering
words_list = list(counts.keys())
# Creating a mapping from words to unique indices
word_index = {w: idx for idx, w in enumerate(words_list)}
# Determining the number of unique words left after filtering
N = len(word_index)

# Creating matrices for storing the Bag of Words representation
X = np.zeros((N_sentences, N))
Y = np.zeros(N_sentences)

i = 0
# Iterating through each sentence again for creating the Bag of Words matrix and assigning populism scores
for index, row in data_mex.iterrows():
    clean_text = row['clean_text'].split()
    gps_Q3_5 = float(row['gps_Q3.5'])  # Ensure this is a float, check for NaNs or missing values
    
    for word in clean_text:
        if word in word_index:
            X[i, word_index[word]] = 1
    
    Y[i] = gps_Q3_5
    i += 1

X = X[:i, :]
Y = Y[:i]


# %% Training (Regression)

random_state = 1
p_train = 0.7
n_splits = 3
n_jobs = 6

# Setup 
np.random.seed(random_state)
indexes = np.random.permutation(len(Y))
n_train = int(p_train * len(Y))
indexes_train = indexes[:n_train]
indexes_test = indexes[n_train:]
X_train, Y_train = X[indexes_train], Y[indexes_train]
X_test, Y_test = X[indexes_test], Y[indexes_test]

# Parameter grid
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [None, 10],
    'min_samples_leaf': [1, 2]
}

best_score = float('inf')
best_params = {}

total_combinations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_leaf']) * n_splits
pbar = tqdm(total=total_combinations)

for n_estimators in param_grid['n_estimators']:
    for max_depth in param_grid['max_depth']:
        for min_samples_leaf in param_grid['min_samples_leaf']:
            scores = []
            for train_index, test_index in KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X_train):
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state, n_jobs=n_jobs)
                model.fit(X_train[train_index], Y_train[train_index])
                predictions = model.predict(X_train[test_index])
                scores.append(mean_squared_error(Y_train[test_index], predictions))
                pbar.update(1)
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}
                best_model = model
                best_score = avg_score  # Corrected the typo from 'pbest_score' to 'best_score'
                best_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}
                best_model = model

pbar.close()

# Saving the best model as a pickle file:
with open('best_random_forest_reg_mx.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Printing scores:
print("Best MSE:", best_score)
print("Best Parameters:", best_params)

# %% Data exploration (leave for now)
