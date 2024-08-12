#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 14:05:14 2024

@author: sarabcidf
"""

# %% Libraries 

import os
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer
from tabulate import tabulate

from IPython.display import display, HTML

# %% WD 

working_directory = '/Users/sarabcidf/Desktop/ASDS/Dissertation/FinalScripts/CorRegAnalysis'
os.chdir(working_directory)

print(f"Current working directory: {os.getcwd()}")

# %% Data

# Reading data (already converted to wide format in R)
gps = pd.read_csv('/Users/sarabcidf/Desktop/ASDS/Dissertation/GlobalPartySurvey/GlobalPartySurvey_Wide_Region.csv')

gps.columns = gps.columns.str.replace('.', '_', regex=False)

# Mapping of old variable names to new descriptive names
variable_mapping = {
    'Q3_5': 'populist_rhetoric',
    'Q3_6': 'populist_saliency',
    'Q5_1': 'will_of_the_people',
    'Q5_2': 'people_should_decide',
    'Q5_3': 'corrupt_politicians',
    'Q5_4': 'strongman_rule',
    'Q3_1': 'ideology'
}

# Rename the columns in the DataFrame
gps.rename(columns=variable_mapping, inplace=True)

# Regions
pop_LA = gps[gps['region'] == "Latin America & Caribbean"].drop(columns='region')
pop_EUR = gps[gps['region'] == "Europe & Central Asia"].drop(columns='region')

# Filtering for Eur and LatAm
gps = gps[gps['region'].isin(["Latin America & Caribbean", "Europe & Central Asia"])]

print(gps.columns) 

# %% Regressions 

gps['region'] = gps['region'].astype('category')

models = [
    smf.ols('populist_rhetoric ~ corrupt_politicians * region + ideology', data=gps).fit(),
    smf.ols('populist_rhetoric ~ people_should_decide * region + ideology', data=gps).fit(),
    smf.ols('populist_rhetoric ~ strongman_rule * region + ideology', data=gps).fit(),
    smf.ols('populist_rhetoric ~ will_of_the_people * region + ideology', data=gps).fit(),
]

# Create a Stargazer object
stargazer = Stargazer(models)

# Customize the output
stargazer.title("Regression Results with Interaction Terms")
stargazer.custom_columns(['Corrupt Politicians', 'People Should Decide', 'Strongman Rule', 'Will of the People'], [1, 1, 1, 1])
stargazer.significant_digits(4)

# Render the table as LaTeX
latex_output = stargazer.render_latex()

# Manually fix the formatting issues
latex_output = latex_output.replace('ˆ', r'\^')
latex_output = latex_output.replace(r"$^{***}$", r"\textsuperscript{***}")
latex_output = latex_output.replace(r"$^{**}$", r"\textsuperscript{**}")
latex_output = latex_output.replace(r"$^{*}$", r"\textsuperscript{*}")
latex_output = latex_output.replace(r"$R^{2}$", r"$R^2$")
latex_output = latex_output.replace(r"Adjusted $R^{2}$", r"Adjusted $R^2$")
latex_output = latex_output.replace(r'p$¡$', r'p$<$')

# Save the LaTeX output to a file
file_path = 'regression_results_with_interactions.tex'
with open(file_path, 'w') as f:
    f.write(latex_output)

# %% Plots

# Mapping for plot labels
plot_labels = {
    'populist_rhetoric': 'Populist Rhetoric',
    'populist_saliency': 'Populist Saliency',
    'will_of_the_people': 'Will of the People',
    'people_should_decide': 'People Should Decide',
    'corrupt_politicians': 'Corrupt Politicians',
    'strongman_rule': 'Strongman Rule'
}

def plot_regression(data, x_var, y_var, region, color, filename):
    plt.figure(figsize=(10, 6))
    ax = sns.regplot(x=x_var, y=y_var, data=data,
                      scatter_kws={'color': 'grey'}, line_kws={'color': color, 'lw': 2}, ci=None)
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                  ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontname('Times New Roman')
        item.set_fontsize(12)
    
    plt.xlabel(plot_labels[x_var], fontsize=13)
    plt.ylabel(plot_labels[y_var], fontsize=13)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
y_var = 'populist_rhetoric'

colors = sns.color_palette("mako", 3)  # Get Mako palette colors

# Plotting and saving plots
variables = ['will_of_the_people', 'people_should_decide', 'corrupt_politicians', 'strongman_rule']
regions = {
    'Overall': (gps, colors[0]),
    'Latin America & Caribbean': (pop_LA, colors[1]),
    'Europe': (pop_EUR, colors[2])
}

for region, (data, color) in regions.items():
    for var in variables:
        filename = f'/Users/sarabcidf/Desktop/ASDS/Dissertation/FinalScripts/CorRegAnalysis/{region.lower().replace(" & ", "_").replace(" ", "_")}_{var}_vs_{y_var}.png'
        plot_regression(data, var, y_var, region, color, filename)


# %% People should decide and corrupt politicians bivariate regressions and plots 

# Just for slope (correlation coefficient)
models2 = [
    smf.ols('populist_rhetoric ~ will_of_the_people', data=pop_LA).fit(),
    smf.ols('populist_saliency ~ corrupt_politicians' , data=pop_LA).fit(),
    smf.ols('populist_rhetoric ~ will_of_the_people', data=pop_EUR).fit(),
    smf.ols('populist_saliency ~ corrupt_politicians', data=pop_EUR).fit()
]

slopes = [model.params[1] for model in models2]

slope_LA_people = slopes[0]
slope_LA_corrupt = slopes[1]
slope_EUR_people = slopes[2]
slope_EUR_corrupt = slopes[3]

p_values = [model.pvalues[1] for model in models2]

def significance_label(p_value):
    if p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    elif p_value < 0.1:
        return '.'
    else:
        return ''
    
p_LA_people = p_values[0]
p_LA_corrupt = p_values[1]
p_EUR_people = p_values[2]
p_EUR_corrupt = p_values[3]

# Plots 

def create_plot(x_variable, xlabel, slope_LA, slope_EUR, p_LA, p_EUR, filename):
    # Colors for scatter and line plots
    colors = sns.color_palette("mako", 3)  # Get Mako palette colors

    # Create the plot with a wider aspect ratio
    plt.figure(figsize=(14, 8))

    # Plot Europe data with triangle markers
    plt.scatter(pop_EUR[x_variable], pop_EUR['populist_rhetoric'], 
                color=colors[2], s=40, marker='^', 
                label=f'Europe - Slope: {slope_EUR:.2f}{significance_label(p_EUR)}')

    # Plot Latin America data with circle markers
    plt.scatter(pop_LA[x_variable], pop_LA['populist_rhetoric'], 
                color=colors[1], s=40, marker='o', 
                label=f'Latin America - Slope: {slope_LA:.2f}{significance_label(p_LA)}')

    # Plot regression lines
    sns.regplot(x=x_variable, y='populist_rhetoric', data=pop_LA, 
                scatter=False, line_kws={'color': colors[1], 'lw': 2}, ci=None)

    sns.regplot(x=x_variable, y='populist_rhetoric', data=pop_EUR, 
                scatter=False, line_kws={'color': colors[2], 'lw': 2}, ci=None)

    # Customize the plot
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel('Populist Rhetoric', fontsize=13)

    # Set font for all text
    for item in ([plt.gca().title, plt.gca().xaxis.label, plt.gca().yaxis.label] +
                 plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        item.set_fontname('Times New Roman')
        item.set_fontsize(12)

    # Add legend and set its font
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
        text.set_fontsize(12)

    # Save plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Create the two plots
create_plot(
    x_variable='corrupt_politicians', 
    xlabel='Corrupt Politicians', 
    slope_LA=slope_LA_corrupt, 
    slope_EUR=slope_EUR_corrupt, 
    p_LA=p_LA_corrupt, 
    p_EUR=p_EUR_corrupt, 
    filename='/Users/sarabcidf/Desktop/ASDS/Dissertation/FinalScripts/CorRegAnalysis/corrupt_politicians_vs_populist_rhetoric_wide.png'
)

create_plot(
    x_variable='people_should_decide', 
    xlabel='People Should Decide', 
    slope_LA=slope_LA_people, 
    slope_EUR=slope_EUR_people, 
    p_LA=p_LA_people, 
    p_EUR=p_EUR_people, 
    filename='/Users/sarabcidf/Desktop/ASDS/Dissertation/FinalScripts/CorRegAnalysis/people_should_decide_vs_populist_rhetoric_wide.png'
)


