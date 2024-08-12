#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 15:04:26 2024

@author: sarabcidf
"""

# %% Libraries and settings

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import joblib
import logging

from collections import Counter
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Suppress verbose output from sklearn
logging.basicConfig(level=logging.WARNING)

print(" -- Libraries done -- ")

# %% WD Setup: (* diff for call *)

def setup_working_directory(working_directory):
    os.chdir(working_directory)
    print("Current working directory:", os.getcwd())

setup_working_directory('/Users/sarabcidf/Desktop/ASDS/Dissertation/FinalScripts/LocalTestsByCountry')

print(" -- WD setup done -- ")

# %% Reading data: (* diff for call *)

data = pd.read_csv('/Users/sarabcidf/Desktop/ASDS/Dissertation/Manifestos/sentences.csv')
data = data.sample(n=2000,random_state=42)

print(" -- Reading data done -- ")

# %% Cleaning data

# Text as string
data['clean_text'] = data['clean_text'].astype(str)

# Dropping unwanted columns
data = data.drop(columns=['md5sum_text', 'url_original', 'mp_total', 'mp_date'])

# Converting columns to defined data types
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

# Removing NAs in text
data = data.dropna(subset=['clean_text'])

print(" -- Cleaning done -- ")

# %% Creating BoW (* changes in Call *)

def create_bow(data):
    N_sentences = len(data)
    word_counts = Counter(word for sentence in data['clean_text'] for word in sentence.split())
    filtered_words = {word for word, count in word_counts.items() if count > 2 and len(word) > 2} # * Change count in Callan * 
    word_index = {word: idx for idx, word in enumerate(filtered_words)}
    N = len(word_index)
    X = lil_matrix((N_sentences, N), dtype=int)
    Y = np.zeros(N_sentences, dtype=np.float32)

    def process_row(row):
        index = row.name
        clean_text = row['clean_text'].split()
        gps_Q3_5 = float(row['gps_Q3.5'])
        
        for word in clean_text:
            if word in word_index:
                X[index, word_index[word]] = 1
        Y[index] = gps_Q3_5

    data = data.reset_index(drop=True)
    data.apply(process_row, axis=1)
    X = X.tocsr()
    return X, Y, word_index

print(" -- Bow F done -- ")

# %% Training and evaluating 

def train_and_evaluate_country(data, country_code):
    country_data = data[data['gps_ISO'] == country_code]
    X, Y, word_index = create_bow(country_data)

    # Train-test split
    random_state = 1
    p_train = 0.7
    indexes_train, indexes_test = train_test_split(np.arange(len(Y)), train_size=p_train, random_state=random_state)
    X_train, Y_train = X[indexes_train], Y[indexes_train]
    X_test, Y_test = X[indexes_test], Y[indexes_test]

    original_test_indices = country_data.iloc[indexes_test].index

    # Parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10],
        'min_samples_leaf': [1, 2]
    }

    # KFold cross-val
    kf = KFold(n_splits=3, shuffle=True, random_state=1)

    # Initializing model
    model = RandomForestRegressor(random_state=1, n_jobs=-1)

    # Initializing GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kf, n_jobs=1, verbose=0)

    # Fitting model
    with joblib.parallel_backend('threading'):
        grid_search.fit(X_train, Y_train)

    # Extracting best parameters and model
    best_score = -grid_search.best_score_
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # print(f"Country: {country_code} - Best Score: {best_score}")
    # print(f"Country: {country_code} - Best Parameters: {best_params}")

    # Evaluating model on test data
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(Y_test, predictions)
    # print(f"Country: {country_code} - Mean Squared Error on Test Data: {mse}")

    return best_model, best_params, best_score, mse, word_index, X_train, X_test, Y_train, Y_test, predictions, original_test_indices

print(" -- Train eval F done -- ")

# %% Loop to process each country

unique_countries = data['gps_ISO'].unique()

results = []
top_features = []

# Adding predcitions as a new column 
data['predictions'] = np.nan

# Processing 
for country in unique_countries:
    print(f"Processing country: {country}")
    model, params, score, mse, word_index, X_train, X_test, Y_train, Y_test, predictions, original_test_indices = train_and_evaluate_country(data, country)
    
    if model is None:
        continue  # Skip if no model is returned

    # Saving best model
    joblib.dump(model, f'best_model_{country}.pkl')
    
    # Saving train/test splits
    np.savez(f'train_test_split_{country}.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)

    results.append({
        'country': country,
        'mse': mse,
        'best_score': score,
        'max_depth': params['max_depth'],
        'min_samples_leaf': params['min_samples_leaf'],
        'n_estimators': params['n_estimators'],
        'model': model,
        'word_index': word_index,
        'X_train': X_train,
        'X_test': X_test,
        'Y_train': Y_train,
        'Y_test': Y_test,
        'predictions': predictions
    })

    # Adding predictions to test data df
    data.loc[original_test_indices, 'predictions'] = predictions

    # Extracting top 3 features
    feature_importances = model.feature_importances_
    top_indices = np.argsort(feature_importances)[-3:][::-1]
    top_words = [list(word_index.keys())[list(word_index.values()).index(i)] for i in top_indices]
    top_features.append({
        'country': country,
        'feature1': top_words[0] if len(top_words) > 0 else '',
        'feature2': top_words[1] if len(top_words) > 1 else '',
        'feature3': top_words[2] if len(top_words) > 2 else ''
    })

# Saving results
results_df = pd.DataFrame(results)
results_df = results_df[['country', 'mse', 'best_score', 'max_depth', 'min_samples_leaf', 'n_estimators']]
results_df.to_csv('model_results.csv', index=False)

# Saving top features
top_features_df = pd.DataFrame(top_features)
top_features_df.to_csv('top_features.csv', index=False)

# Saving data with predictions
data.to_csv('data_with_predictions.csv', index=False)

# Saving detailed results as txt
with open('model_results.txt', 'w') as f:
    for result in results:
        f.write(f"Country: {result['country']}\n")
        f.write(f"Best Parameters: max_depth={result['max_depth']}, min_samples_leaf={result['min_samples_leaf']}, n_estimators={result['n_estimators']}\n")
        f.write(f"Best Score: {result['best_score']}\n")
        f.write(f"Mean Squared Error: {result['mse']}\n\n")

print(" -- Processing done -- ")

# %% Plotting results (* font available in Callan *)

# Apply settings for plots
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Nimbus Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
sns.set(style="whitegrid", palette='mako')

iso_to_country = {
    'BOL': 'Bolivia',
    'COL': 'Colombia',
    'CRI': 'Costa Rica',
    'ECU': 'Ecuador',
    'CHL': 'Chile',
    'PAN': 'Panama',
    'URY': 'Uruguay',
    'DOM': 'D. Republic',
    'MEX': 'Mexico',
    'PER': 'Peru'
}

def plot_results(country_code, results):
    country_name = iso_to_country[country_code]
    result = next(item for item in results if item['country'] == country_code)
    model = result['model']
    word_index = result['word_index']
    X_train = result['X_train']
    X_test = result['X_test']
    Y_train = result['Y_train']
    Y_test = result['Y_test']

    # Feature importance
    feature_importances = model.feature_importances_
    top_indices = np.argsort(feature_importances)[-10:][::-1]
    top_words = [list(word_index.keys())[list(word_index.values()).index(i)] for i in top_indices]
    top_importances = feature_importances[top_indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_words, palette='mako')
    plt.xlabel('Feature Importance', fontsize=12, family='Nimbus Roman')
    plt.ylabel('Words', fontsize=12, family='Nimbus Roman')
    plt.title(f'Top 10 Features for {country_name}', fontsize=14, family='Nimbus Roman')
    plt.xticks(fontsize=10, family='Nimbus Roman')
    plt.yticks(fontsize=10, family='Nimbus Roman')
    plt.savefig(f'{country_name}_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Error rate (train and test)
    train_errors = []
    test_errors = []
    colors = sns.color_palette('mako', n_colors=2)
    for n in range(1, result['n_estimators'] + 1):
        partial_model = RandomForestRegressor(n_estimators=n, max_depth=result['max_depth'], min_samples_leaf=result['min_samples_leaf'], random_state=1, n_jobs=-1)
        partial_model.fit(X_train, Y_train)
        train_pred = partial_model.predict(X_train)
        test_pred = partial_model.predict(X_test)
        train_errors.append(mean_squared_error(Y_train, train_pred))
        test_errors.append(mean_squared_error(Y_test, test_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, result['n_estimators'] + 1), train_errors, label='Train Error', color=colors[0])
    plt.plot(range(1, result['n_estimators'] + 1), test_errors, label='Test Error', color=colors[1])
    plt.xlabel('Number of Trees', fontsize=12, family='Nimbus Roman')
    plt.ylabel('Mean Squared Error', fontsize=12, family='Nimbus Roman')
    plt.title(f'Error Rate for {country_name}', fontsize=14, family='Nimbus Roman')
    plt.legend(prop={'family': 'Nimbus Roman'})
    plt.xticks(fontsize=10, family='Nimbus Roman')
    plt.yticks(fontsize=10, family='Nimbus Roman')
    plt.savefig(f'{country_name}_error_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Learning curves
    train_sizes = np.linspace(0.1, 0.9, 9)
    train_scores = []
    validation_scores = []

    for train_size in train_sizes:
        X_train_partial, _, Y_train_partial, _ = train_test_split(X_train, Y_train, train_size=train_size, random_state=1)
        partial_model = RandomForestRegressor(n_estimators=result['n_estimators'], max_depth=result['max_depth'], min_samples_leaf=result['min_samples_leaf'], random_state=1, n_jobs=-1)
        partial_model.fit(X_train_partial, Y_train_partial)
        train_pred = partial_model.predict(X_train_partial)
        validation_pred = partial_model.predict(X_test)
        train_scores.append(mean_squared_error(Y_train_partial, train_pred))
        validation_scores.append(mean_squared_error(Y_test, validation_pred))

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes * 100, train_scores, label='Train Error', color=colors[0])
    plt.plot(train_sizes * 100, validation_scores, label='Validation Error', color=colors[1])
    plt.xlabel('Training Set Size (%)', fontsize=12, family='Nimbus Roman')
    plt.ylabel('Mean Squared Error', fontsize=12, family='Nimbus Roman')
    plt.title(f'Learning Curve for {country_name}', fontsize=14, family='Nimbus Roman')
    plt.legend(prop={'family': 'Nimbus Roman'})
    plt.xticks(fontsize=10, family='Nimbus Roman')
    plt.yticks(fontsize=10, family='Nimbus Roman')
    plt.savefig(f'{country_name}_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

# Loop to plot for each country
for country in unique_countries:
    plot_results(country, results)

