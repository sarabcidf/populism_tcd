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

# %% Summary stats

data
data.head()
data.columns 
print(data.iloc[0])
    
## Countries 

data['gps_ISO'].nunique()
data['gps_ISO'].dropna().unique()

## Sentences by country: 
    
# Configuring matplotlib to use a serif font similar to LaTeX's Computer Modern
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'  # Or another similar serif font
mpl.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts look somewhat like LaTeX's fonts

# Setting the visual style for the plots
sns.set(style="whitegrid")

## Calculating the percentage of sentences for each country
country_percentages = data['gps_ISO'].value_counts(normalize=True) * 100

## Plotting
plt.figure(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(country_percentages)))  # Create a color range based on the number of bars
country_percentages.sort_values(ascending=False).plot(kind='barh', color=colors)  # Apply colors to each bar
plt.title('Percentage of Sentences by Country', fontsize=14)
plt.xlabel('Percentage', fontsize=12)
plt.ylabel('Country ISO Codes', fontsize=12)

# Showing the plot
plt.show()

# Saving
plt.savefig('country_percentage_distribution.png', format='png', dpi=300, bbox_inches='tight')

# Getting rid of NAs in target: 
print("NaNs:", data['clean_text'].isna().sum())
data['clean_text'] = data['clean_text'].fillna('')

# %% Creating BoW (authors 00)

# CREATING SAMPLE # REMOVE AFTERWARDS ***

data = data.sample(n=50000, random_state=1)

# Calculating the total number of sentences in the dataset
N_sentences = len(data)

# Initializing a dictionary to count occurrences of each word
counts = defaultdict(int)

# Iterating through each sentence to count word frequencies
for index, row in data.iterrows():
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
for index, row in data.iterrows():
    clean_text = row['clean_text'].split()
    gps_Q3_5 = float(row['gps_Q3.5'])
    gps_Q3_1 = float(row['gps_Q3.1'])
    
    for word in clean_text:
        if word in word_index:
            X[i, word_index[word]] = 1

    # Determining if a party is populist based on gps_Q3.5
    Y[i] = 1 if gps_Q3_5 > 5 else 0
    # Determining political orientation based on gps_Q3.1
    orientation = "left" if gps_Q3_1 <= 5 else "right" if gps_Q3_1 > 5 else "other"
    orientations.append(orientation)
    i += 1

# Reducing the size of matrices to the actual number of processed sentences
X = X[:i, :]
Y = Y[:i]
orientations = np.array(orientations)

# %% Exploring the results 

# Calculating counts and percentages for populist vs non-populist labels:
total_sentences = len(Y)
populist_count = np.sum(Y == 1)
non_populist_count = np.sum(Y == 0)
populist_percentage = populist_count / total_sentences * 100
non_populist_percentage = non_populist_count / total_sentences * 100

# Bar chart for Distribution of Populist and Non-Populist Sentences (percentages)
plt.figure(figsize=(12, 6))
plt.bar(['Populist', 'Non-Populist'], [populist_percentage, non_populist_percentage], color=['blue', 'red'])
plt.title('Percentage of Populist and Non-Populist Sentences', fontsize=14)
plt.ylabel('Percentage of Sentences', fontsize=12)
plt.savefig('populist_non_populist_percentage_distribution.png', format='png', dpi=300, bbox_inches='tight')  # Save as PNG
plt.show()

# Distribution of orientations
unique, counts = np.unique(orientations, return_counts=True)

# Bar chart for Distribution of Political Orientations (percentages)
plt.figure(figsize=(12, 6))
orientation_colors = plt.cm.viridis(np.linspace(0, 1, len(unique)))  # Generate colors for each unique orientation
plt.bar(unique, counts / total_sentences * 100, color=orientation_colors)
plt.title('Distribution of Political Orientations', fontsize=14)
plt.ylabel('Percentage of Sentences', fontsize=12)
plt.savefig('political_orientations_percentage_distribution.png', format='png', dpi=300, bbox_inches='tight')  # Save as PNG
plt.show()

# %% Training (authors 01 modified)

# nation="IT" (I don't need this one, I already have just Mx plus all countries are Spanish)

model_type = "LogisticRegression"
target_score = "AUC"
n_splits = 3
p_train = 0.7
random_state = 1
n_jobs = 4

np.random.seed(random_state)
indexes = np.random.permutation(len(Y))
n_train = int(p_train * len(Y))
indexes_train = indexes[:n_train]
indexes_test = indexes[n_train:]
X_train, Y_train = X[indexes_train], Y[indexes_train]
X_test, Y_test = X[indexes_test], Y[indexes_test]

param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2']
}

best_score = 0
best_params = {}
pbar = tqdm(total=len(param_grid['C']) * len(param_grid['penalty']) * n_splits)

for C in param_grid['C']:
    for penalty in param_grid['penalty']:
        scores = []
        for train_index, test_index in KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X_train):
            model = LogisticRegression(C=C, penalty=penalty, random_state=random_state, solver='lbfgs')
            model.fit(X_train[train_index], Y_train[train_index])
            pred = model.predict_proba(X_train[test_index])[:, 1]
            scores.append(roc_auc_score(Y_train[test_index], pred))
            pbar.update(1)
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = {'C': C, 'penalty': penalty}
            best_model = model

pbar.close()

with open('best_logistic_regression_model_all_bin.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("Best Score:", best_score)
print("Best Parameters:", best_params)

