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

from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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

# Ensuring 'clean_text' column is of type string
data['clean_text'] = data['clean_text'].astype(str)

# Dropping unwanted columns
data = data.drop(columns=['md5sum_text', 'url_original', 'mp_total', 'mp_date'])

# Converting columns to appropriate data types
categorical_columns = [
    'manifesto_id', 'party', 'language', 'source', 'pf_id', 'mp_id', 
    'mp_partyname', 'mp_partyabbrev', 'mp_parfam', 'mp_progtype', 
    'mp_corpusversion', 'mp_datasetversion', 'mp_id_perm', 
    'pf_name_short', 'pf_name', 'pf_name_english', 'gps_ISO', 
    'gps_PartyName', 'gps_PartyAbb'
]
for col in categorical_columns:
    data[col] = data[col].astype('category')

float_columns = [
    'pf_share_year', 'gps_Q3.1', 'gps_Q3.2', 'gps_Q3.3', 
    'gps_Q3.4', 'gps_Q3.5', 'gps_Q3.6', 'gps_Q5.1', 
    'gps_Q5.2', 'gps_Q5.3', 'gps_Q5.4'
]
for col in float_columns:
    data[col] = data[col].astype('float32')

# Getting rid of NAs in target
def check_na(text):
    if 'na' in text.split():
        return np.nan
    return text

data['clean_text'] = data['clean_text'].apply(check_na)

# Removing rows with NaN values in 'clean_text'
data = data.dropna(subset=['clean_text'])

print(" -- Cleaning done -- ")

# %% Creating TF-IDF

# Function to create TF-IDF representation
def create_tfidf(data):
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.95, max_features=10000)
    X = tfidf_vectorizer.fit_transform(data['clean_text'])
    Y = data['gps_Q3.5'].astype(np.float32).values
    return X, Y

# %% Function to train and evaluate model for each country
def train_and_evaluate_country(data, country_code):
    country_data = data[data['gps_ISO'] == country_code].sample(n=1000, random_state=42)
    X, Y = create_tfidf(country_data)

    # Train-test split
    random_state = 1
    p_train = 0.7
    indexes_train, indexes_test = train_test_split(np.arange(len(Y)), train_size=p_train, random_state=random_state)
    X_train, Y_train = X[indexes_train], Y[indexes_train]
    X_test, Y_test = X[indexes_test], Y[indexes_test]

    # Parameter grid
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [None, 10],
        'min_samples_leaf': [1, 2]
    }

    # Setting up KFold cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=1)

    # Initializing the model
    model = RandomForestRegressor(random_state=1, n_jobs=-1)

    # Initializing GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1, verbose=2)

    # Fitting the model
    grid_search.fit(X_train, Y_train)

    # Extracting the best parameters and model
    best_score = -grid_search.best_score_
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print(f"Country: {country_code} - Best Score: {best_score}")
    print(f"Country: {country_code} - Best Parameters: {best_params}")

    # Evaluating model on test data
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(Y_test, predictions)
    print(f"Country: {country_code} - Mean Squared Error on Test Data: {mse}")

    return best_model, best_params, best_score, mse

# %% Main loop to process each country
unique_countries = data['gps_ISO'].unique()

results = []
for country in unique_countries:
    print(f"Processing country: {country}")
    model, params, score, mse = train_and_evaluate_country(data, country)
    results.append({
        'country': country,
        'best_params': params,
        'best_score': score,
        'mse': mse
    })

# Converting results to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('country_results.csv', index=False)

print(" -- Processing done -- ")
