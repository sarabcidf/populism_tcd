# -*- coding: utf-8 -*-

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


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier

# pip install nltk
# pip install tqdm
# nltk.download('punkt')

# %% WD

working_directory = '/Users/sarabcidf/Desktop/ASDS/Dissertation/Python'
os.chdir(working_directory)
print("Current working directory:", os.getcwd())

# %% Reading data

data = pd.read_csv('/Users/sarabcidf/Desktop/ASDS/Dissertation/Manifestos/sentences.csv')

# %% Exploring and cleaning

data
data.head()
data.columns 
print(data.iloc[0])

# Getting rid of NAs in target: 

print("NaNs:", data['clean_text'].isna().sum())
data['clean_text'] = data['clean_text'].fillna('')

# %% Filtering for Bolivia

np.unique(data['gps_ISO'])
data_col = data[data['gps_ISO'] == 'BOL']

# %% Creating BoW (authors 00)

# Calculating the total number of sentences in the dataset
N_sentences = len(data_col)

# Initializing a dictionary to count occurrences of each word
counts = defaultdict(int)

# Iterating through each sentence to count word frequencies
for index, row in data_col.iterrows():
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

# Creating matrices for storing the Bag of Words representation and labels
X = np.zeros((N_sentences, N))
Y = np.zeros(N_sentences)
# Initializing a list to store political orientations
orientations = []
i = 0

# Iterating through each sentence again for creating the Bag of Words matrix and assigning labels
for index, row in data_col.iterrows():
    clean_text = row['clean_text'].split()
    gps_Q3_5 = float(row['gps_Q3.5'])
    gps_Q3_1 = float(row['gps_Q3.1'])
    
    for word in clean_text:
        if word in word_index:
            X[i, word_index[word]] = 1

    # Directly determining if a party is populist based on gps_Q3.5
    Y[i] = 1 if gps_Q3_5 > 5 else 0
    # Determining political orientation based on gps_Q3.1
    orientation = "left" if gps_Q3_1 <= 5 else "right" if gps_Q3_1 > 5 else "other"
    orientations.append(orientation)
    i += 1

# Reduce the size of matrices to the actual number of processed sentences
X = X[:i, :]
Y = Y[:i]
orientations = np.array(orientations)

# %% Exploring the results 

# Check the balance of populist vs non-populist labels
populist_count = np.sum(Y == 1)
non_populist_count = np.sum(Y == 0)

print("Number of populist sentences:", populist_count)
print("Number of non-populist sentences:", non_populist_count)

# Bar chart for Populist vs Non-Populist
plt.bar(['Populist', 'Non-Populist'], [populist_count, non_populist_count], color=['blue', 'red'])
plt.title('Distribution of Populist and Non-Populist Sentences')
plt.ylabel('Number of Sentences')
plt.show()

# Distribution of orientations
unique, counts = np.unique(orientations, return_counts=True)
plt.bar(unique, counts, color=['green', 'orange', 'gray'])
plt.title('Distribution of Political Orientations')
plt.ylabel('Number of Sentences')
plt.show()

# %% Training (authors 01 modified)

# nation="IT" (I don't need this one, I already have just Mx plus all countries are Spanish)

model_type = "RandomForest"
target_score = "AUC"
n_splits = 3 # Authors originally use 5
p_train = 0.7
random_state = 1
n_jobs = 6 # Authors originally use 8 (I only got 8 total)

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
    'n_estimators': [100, 150], # Authors originally use 100, 200
    'max_depth': [None, 10], # Authors originally use None, 10, 20
    'min_samples_leaf': [1, 2] # Authors originally use 1, 2, 4
}

# Initializing variables for tracking the best score and parameters
best_score = 0
best_params = {}

# Setting up progress bar
total_combinations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_leaf']) * n_splits
pbar = tqdm(total=total_combinations)

# Nested loop to handle parameter grid search
for n_estimators in param_grid['n_estimators']:
    for max_depth in param_grid['max_depth']:
        for min_samples_leaf in param_grid['min_samples_leaf']:
            scores = []
            for train_index, test_index in KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X_train):
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf, random_state=random_state, n_jobs=n_jobs)
                model.fit(X_train[train_index], Y_train[train_index])
                pred = model.predict_proba(X_train[test_index])[:, 1]
                scores.append(roc_auc_score(Y_train[test_index], pred))
                pbar.update(1)
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}
                best_model = model  # Keep track of the best model

pbar.close()

# Saving best model as pickle file:
with open('best_random_forest_model_bol.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Printing scores: 
print("Best Score:", best_score)
print("Best Parameters:", best_params)
