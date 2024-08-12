#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 12:29:20 2024

@author: sarabcidf
"""

# %% Libraries

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

print(" -- Libraries done -- ")


# %% WD Setup: (* diff for call *)

working_directory = '/Users/sarabcidf/Desktop/ASDS/Dissertation/FinalScripts/SummaryStats'
os.chdir(working_directory)
print("Current working directory:", os.getcwd())

print(" -- WD done -- ")


# %% Reading data: (* diff for call *)

data = pd.read_csv('/Users/sarabcidf/Desktop/ASDS/Dissertation/Manifestos/sentences.csv')

print(" -- Reading data done -- ")


# %% Brief exploring and cleaning data:
    
# Quick look (* do not run on call *)
data.head(5)
data.columns 
print(data.iloc[0])

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

# Print all unique gps_ISO values
print(data['gps_ISO'].dropna().unique())

# %% Mapping ISO codes to full country names

# Create the mapping dictionary
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

# Apply the mapping to create a new column with full country names
data['country_name'] = data['gps_ISO'].map(iso_to_country)

# %% More exploring for summary stats 

## Settings to plot
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 12

# Setting the visual style for the plots
sns.set(style="whitegrid")

# Plotting function to ensure consistent styling
def plot_bar(data, x, y, title, xlabel, ylabel, filename, rotation=45):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x, y=y, palette='mako')
    plt.title(title, fontsize=14, fontname='Times New Roman')
    plt.xlabel(xlabel, fontsize=12, fontname='Times New Roman')
    plt.ylabel(ylabel, fontsize=12, fontname='Times New Roman')
    plt.xticks(rotation=rotation, fontsize=10, fontname='Times New Roman')
    plt.yticks(fontsize=10, fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

## Countries

# Number of manifestos by country
manifesto_counts_by_country = data.groupby('country_name')['manifesto_id'].nunique().sort_values(ascending=False)
plot_bar(manifesto_counts_by_country, manifesto_counts_by_country.index, manifesto_counts_by_country.values,
         'Number of Manifestos by Country', 'Country', 'Number of Manifestos', 'manif_by_country.png')

# Number of sentences by country
sentences_by_country = data['country_name'].value_counts().sort_values(ascending=False)
plot_bar(sentences_by_country, sentences_by_country.index, sentences_by_country.values,
         'Number of Sentences by Country', 'Country', 'Number of Sentences', 'sent_by_country.png')

# Number of parties by country
parties_by_country = data.groupby('country_name')['pf_name_english'].nunique().sort_values(ascending=False)
plot_bar(parties_by_country, parties_by_country.index, parties_by_country.values,
         'Number of Parties by Country', 'Country', 'Number of Parties', 'parties_by_country.png')

## Populism score

data['populism_score_bin'] = pd.cut(data['gps_Q3.5'], bins=list(range(0, 11)), right=False)

# Number of sentences by populism score
sentences_by_populism = data['populism_score_bin'].value_counts().sort_index()
plot_bar(sentences_by_populism, sentences_by_populism.index.astype(str), sentences_by_populism.values,
         'Number of Sentences by Populism Score', 'Populism Score Interval', 'Number of Sentences', 'sent_by_pop_score.png')

# Number of manifestos by populism score
manifestos_by_populism = data.groupby('populism_score_bin')['manifesto_id'].nunique()
plot_bar(manifestos_by_populism, manifestos_by_populism.index.astype(str), manifestos_by_populism.values,
         'Number of Manifestos by Populism Score', 'Populism Score Interval', 'Number of Manifestos', 'manif_by_pop_score.png')

# Number of parties by populism score
parties_by_populism = data.groupby('populism_score_bin')['pf_name_english'].nunique()
plot_bar(parties_by_populism, parties_by_populism.index.astype(str), parties_by_populism.values,
         'Number of Parties by Populism Score', 'Populism Score Interval', 'Number of Parties', 'parties_by_pop_score.png')

## Ideology

data['ideology'] = data['gps_Q3.5'].apply(lambda x: 'Left' if x <= 5 else 'Right')

# Number of sentences by ideology
sentences_by_ideology = data['ideology'].value_counts()
plot_bar(sentences_by_ideology, sentences_by_ideology.index, sentences_by_ideology.values,
         'Number of Sentences by Political Ideology', 'Political Ideology', 'Number of Sentences', 'sent_by_ideology.png')

# Number of manifestos by ideology
manifestos_by_ideology = data.groupby('ideology')['manifesto_id'].nunique()
plot_bar(manifestos_by_ideology, manifestos_by_ideology.index, manifestos_by_ideology.values,
         'Number of Manifestos by Political Ideology', 'Political Ideology', 'Number of Manifestos', 'manif_by_ideology.png')

# Number of parties by ideology
parties_by_ideology = data.groupby('ideology')['pf_name_english'].nunique()
plot_bar(parties_by_ideology, parties_by_ideology.index, parties_by_ideology.values,
         'Number of Parties by Political Ideology', 'Political Ideology', 'Number of Parties', 'parties_by_ideology.png')


# Simple counts
total_sentences = data.shape[0]
total_manifestos = data['manifesto_id'].nunique()
total_parties = data['pf_name_english'].nunique()
total_countries = data['gps_ISO'].nunique()

print(f"Total number of sentences: {total_sentences}")
print(f"Total number of manifestos: {total_manifestos}")
print(f"Total number of parties: {total_parties}")
print(f"Total number of countries: {total_countries}")