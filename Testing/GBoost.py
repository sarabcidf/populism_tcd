#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 15:04:26 2024

@author: sarabcidf
"""

# -*- coding: utf-8 -*-

# %% Libraries

import numpy as np
import pandas as pd
import os
import pickle

from collections import Counter
from scipy.sparse import lil_matrix, save_npz
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor  # Changed import

print(" -- Libraries done -- ")


# %% WD Setup: (* diff for call *)

working_directory = '/Users/sarabcidf/Desktop/Testing'
os.chdir(working_directory)
print("Current working directory:", os.getcwd())

print(" -- WD done -- ")


# %% Reading data: (* diff for call *)

data = pd.read_csv('/Users/sarabcidf/Desktop/ASDS/Dissertation/Manifestos/sentences.csv')

print(" -- Reading data done -- ")


# %% Brief exploring and cleaning data:

# Sample to test on local machine: (* do not run on call *)
data = data.sample(n=1000, random_state=42)
    
# Quick look (* do not run on call *)
data
data.head(5)
data.columns 
print(data.iloc[0])

# Specifying all types for efficiency

print(data.dtypes) # (* do not run on call *)

# To category
data['manifesto_id'] = data['manifesto_id'].astype('category')
data['party'] = data['party'].astype('category')
data['language'] = data['language'].astype('category')
data['source'] = data['source'].astype('category')
data['pf_id'] = data['pf_id'].astype('category')
data['mp_id'] = data['mp_id'].astype('category')
data['mp_partyname'] = data['mp_partyname'].astype('category')
data['mp_partyabbrev'] = data['mp_partyabbrev'].astype('category')
data['mp_parfam'] = data['mp_parfam'].astype('category')
data['mp_progtype'] = data['mp_progtype'].astype('category')
data['mp_corpusversion'] = data['mp_corpusversion'].astype('category')
data['mp_datasetversion'] = data['mp_datasetversion'].astype('category')
data['mp_id_perm'] = data['mp_id_perm'].astype('category')
data['pf_name_short'] = data['pf_name_short'].astype('category')
data['pf_name'] = data['pf_name'].astype('category')
data['pf_name_english'] = data['pf_name_english'].astype('category')
data['gps_ISO'] = data['gps_ISO'].astype('category')
data['gps_PartyName'] = data['gps_PartyName'].astype('category')
data['gps_PartyAbb'] = data['gps_PartyAbb'].astype('category')

# To smaller floats
data['pf_share_year'] = data['pf_share_year'].astype('float32')
data['gps_Q3.1'] = data['gps_Q3.1'].astype('float32')
data['gps_Q3.2'] = data['gps_Q3.2'].astype('float32')
data['gps_Q3.3'] = data['gps_Q3.3'].astype('float32')
data['gps_Q3.4'] = data['gps_Q3.4'].astype('float32')
data['gps_Q3.5'] = data['gps_Q3.5'].astype('float32')
data['gps_Q3.6'] = data['gps_Q3.6'].astype('float32')
data['gps_Q5.1'] = data['gps_Q5.1'].astype('float32')
data['gps_Q5.2'] = data['gps_Q5.2'].astype('float32')
data['gps_Q5.3'] = data['gps_Q5.3'].astype('float32')
data['gps_Q5.4'] = data['gps_Q5.4'].astype('float32')

data = data.drop(columns=['md5sum_text', 'url_original', 'mp_total', 'mp_date'])

# Getting rid of NAs in target: 
def check_na(text):
    if 'na' in text.split():
        return np.nan
    return text

data['clean_text'] = data['clean_text'].apply(check_na)

# Ensuring 'clean_text' column is of type string
data['clean_text'] = data['clean_text'].astype(str)

# Saving filtered and/or sampled data: 
data.to_csv('data.csv', index=False)

print(" -- Cleaning done -- ")


# %% Creating BoW

# Calculating the total number of sentences in the dataset
N_sentences = len(data)

# Counting occurrences of each word using Counter
word_counts = Counter(word for sentence in data['clean_text'] for word in sentence.split())

# Filtering out infrequent and short words
filtered_words = {word for word, count in word_counts.items() if count > 4 and len(word) > 2}

# Creating a mapping from words to unique indices
word_index = {word: idx for idx, word in enumerate(filtered_words)}

# Determining the number of unique words left after filtering
N = len(word_index)

# Creating sparse matrices for storing the Bag of Words representation
X = lil_matrix((N_sentences, N), dtype=int)
Y = np.zeros(N_sentences, dtype=np.float32)

# Function to process each row
def process_row(row):
    index = row.name
    clean_text = row['clean_text'].split()
    gps_Q3_5 = float(row['gps_Q3.5'])
    
    for word in clean_text:
        if word in word_index:
            X[index, word_index[word]] = 1
    Y[index] = gps_Q3_5

# Ensuring DataFrame has a range index
data = data.reset_index(drop=True)

# Applying the function to each row
data.apply(process_row, axis=1)

# Converting X to CSR format for efficient row slicing operations
X = X.tocsr()
print(" -- BoW done -- ")

# Save the BoW representation to disk
save_npz('X.npz', X)
np.save('Y.npy', Y)

print(" -- BoW done -- ")

# %% Train-test split

# Setup and split
random_state = 1
p_train = 0.7
indexes_train, indexes_test = train_test_split(np.arange(N_sentences), train_size=p_train, random_state=random_state)
X_train, Y_train = X[indexes_train], Y[indexes_train]
X_test, Y_test = X[indexes_test], Y[indexes_test]

# Saving train and test data
save_npz('X_train.npz', X_train)
np.save('Y_train.npy', Y_train)
save_npz('X_test.npz', X_test)
np.save('Y_test.npy', Y_test)
np.save('test_indices.npy', indexes_test)

print(" -- Split done -- ")


# %% Training Gradient Boosting Regressor

# Parameter grid
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# Setting up KFold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=1)

# Initializing the model
model = GradientBoostingRegressor(random_state=1)

# Initializing GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1, verbose=2)

# Fitting the model
grid_search.fit(X_train, Y_train)

# Extracting the best parameters and model
best_score = -grid_search.best_score_
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best Score: {best_score}")
print(f"Best Parameters: {best_params}")
print(" -- Training done -- ")

# Saving the best model using pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(" -- Model saved -- ")

# %% Libraries

import numpy as np
import pandas as pd

import os
import pickle

from scipy.sparse import load_npz

from sklearn.metrics import mean_squared_error

# %% Reading pkl : 

# Load the pickled model 
with open('best_model.pkl', 'rb') as file:
     best_model = pickle.load(file)
    
# %% Re-loading data and splits: 

# Data
data = pd.read_csv('data.csv')

# Splits 
X_train = load_npz('X_train.npz')
X_test = load_npz('X_test.npz')
Y_train = np.load('Y_train.npy')
Y_test = np.load('Y_test.npy')

# Indexes
indexes_test = np.load('test_indices.npy')

# Test data

test_data = data.loc[indexes_test]

# %% Exploring model results

predictions = best_model.predict(X_test)
test_data['predicted_score'] = predictions

# Looking at (average) predicted scores by party
# Grouping by the party and calculate the mean predicted score
party_avg_scores = test_data.groupby('pf_name')['predicted_score'].mean().reset_index()
party_avg_scores = party_avg_scores.sort_values(by='predicted_score', ascending=False)

# Display the aggregated scores
print(party_avg_scores)

# Best parameters
print("Best Parameters:", best_model.get_params())

# Evaluating model on test data
predictions = best_model.predict(X_test)
mse = mean_squared_error(Y_test, predictions)
print("Mean Squared Error on Test Data:", mse)
