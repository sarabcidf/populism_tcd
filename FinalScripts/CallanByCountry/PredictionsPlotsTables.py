#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:46:24 2024

@author: sarabcidf
"""

# %% Libraries

import numpy as np
import pandas as pd
import os

# WD

def setup_working_directory(working_directory):
    os.chdir(working_directory)
    print("Current working directory:", os.getcwd())

setup_working_directory('/Users/sarabcidf/Desktop/ASDS/Dissertation/FinalScripts/CallanByCountry')

# %% Data with predictions

# Loading data with predictions
data = pd.read_csv('data_with_predictions.csv')

print(data.columns)

# Removing NA in predictions (keeping test data) 
data = data.dropna(subset=['predictions'])

# Adding country name
data['country'] = data['gps_ISO'].map({
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
})

# %% Standardizing and scaling predictions

# Function to scale/adjust predictions
def adjust_predictions(predictions, target_mean, target_std, target_min, target_max):
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    adjusted_predictions = (predictions - pred_mean) / pred_std * target_std + target_mean
    # Clip predictions to the target range
    adjusted_predictions = np.clip(adjusted_predictions, target_min, target_max)
    return adjusted_predictions

# Defining the range and statistics of target
target_min, target_max = 1.5, 8.6
target_mean = np.mean(data['predictions'])
target_std = np.std(data['predictions'])

# Adjusting
data['adjusted_predictions'] = adjust_predictions(data['predictions'], target_mean, target_std, target_min, target_max)

# %% Average predictions by party 

# Calculate the average adjusted predicted score by party and include the country
avg_score_party = data.groupby(['mp_partyname', 'country'])['adjusted_predictions'].mean().reset_index()
avg_score_party = avg_score_party.sort_values(by='adjusted_predictions', ascending=False)

# Spanish
avg_score_party_es = data.groupby(['pf_name', 'country'])['adjusted_predictions'].mean().reset_index()
avg_score_party_es = avg_score_party_es.sort_values(by='adjusted_predictions', ascending=False)

# Displaying results
print(avg_score_party)
print(avg_score_party_es)

# Saving as CSV table
avg_score_party.to_csv('avg_score_party.csv', index=False)

# Parties into bins
party_bins = pd.qcut(avg_score_party['adjusted_predictions'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Sampling 2 parties per bin
sampled_parties = avg_score_party.groupby(party_bins).apply(lambda x: x.sample(2, random_state=1)).reset_index(drop=True)
sampled_parties = sampled_parties.sort_values(by='adjusted_predictions', ascending=False)

# Displaying results
print(sampled_parties)

# Saving as CSV table
sampled_parties.to_csv('sampled_parties.csv', index=False)

# %% Predictions for a sample of sentences 

# Dividing data into bins
bins = pd.qcut(data['adjusted_predictions'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Sampling 2 sentences per bin
sampled_sentences = data.groupby(bins).apply(lambda x: x.sample(2, random_state=1)).reset_index(drop=True)

# Selecting only relevant columns
sampled_sentences = sampled_sentences[['clean_text', 'adjusted_predictions', 'country', 'mp_partyname']]

# Ordering
sampled_sentences = sampled_sentences.sort_values(by='adjusted_predictions', ascending=False)

# Displaying results
print(sampled_sentences)

sampled_sentences.to_csv('sampled_sentences.csv', index=False)
