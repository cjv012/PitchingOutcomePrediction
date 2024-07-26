import numpy as np
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve


from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from keras.layers import Dense, Activation
from keras.optimizers import Adam, SGD
from scikeras.wrappers import KerasClassifier


df_pitches = pd.read_csv('data/savant_data (2).csv')
for i in range(8):
    x = i+3
    df_new_rows = pd.read_csv('data/savant_data (' + str(x) + ').csv')
    df_pitches = pd.concat([df_pitches, df_new_rows])
# Set the index of the DataFrame to the 'player_name' column to allow for easier data access by player name

 # Create a copy of the df_pitches DataFrame to clean and manipulate the data without altering the original data
df_pitches_cleaned = df_pitches
# Replace occurrences of 'hit_into_play' in the 'description' column with NaN (not a number) values
df_pitches_cleaned['description'].replace('hit_into_play', np.nan, inplace=True)
# Fill any NaN values in the 'description' column with corresponding values from the 'events' column
df_pitches_cleaned['description'] = df_pitches_cleaned['description'].fillna(df_pitches_cleaned['events'])
# Drop the 'events' column from the DataFrame as it is no longer needed after merging its data into 'description'
df_pitches_cleaned.drop('events', axis=1, inplace=True)
# Display the first 10 rows of the cleaned DataFrame to verify changes and see the cleaned data
 # Create a new DataFrame 'df_pitches_compact' by selecting specific columns from 'df_pitches_cleaned'
df_pitches_compact = df_pitches_cleaned[['pitch_type', 'release_speed', 'effective_speed', 'spin_axis', 'release_spin_rate', 'release_extension', 'p_throws', 'pfx_x', 'pfx_z', 'description']]
# Rename the 'release_spin_rate' column to 'spin_rate' for clarity and simplicity
df_pitches_compact = df_pitches_compact.rename(columns={'release_spin_rate': 'spin_rate'})
# Rename the 'p_throws' column to 'pitcher_handedness' to more clearly describe the data it represents
df_pitches_compact = df_pitches_compact.rename(columns={'p_throws': 'pitcher_handedness'})
# Rename the 'pfx_x' column to 'horizontal_movement (ft)' to clarify that it represents the horizontal movement of the pitch in feet
df_pitches_compact = df_pitches_compact.rename(columns={'pfx_x': 'horizontal_movement (ft)'})
# Rename the 'pfx_z' column to 'vertical_movement (ft)' to clarify that it represents the vertical movement of the pitch in feet
df_pitches_compact = df_pitches_compact.rename(columns={'pfx_z': 'vertical_movement (ft)'})
# Display the first few rows of the updated DataFrame to check the new structure and the changes made
 # convert necessary columns to categorical
df_pitches_compact['pitch_type'] = pd.Categorical(df_pitches_compact['pitch_type'])
df_pitches_compact['pitcher_handedness'] = pd.Categorical(df_pitches_compact['pitcher_handedness'])
df_pitches_compact['description'] = pd.Categorical(df_pitches_compact['description'])

# Convert all numeric variables to simplest numeric types
df_float_downcast = df_pitches_compact.select_dtypes('float').columns
df_pitches_compact[df_float_downcast]=df_pitches_compact[df_float_downcast].apply(pd.to_numeric, downcast='float')
df_pitches_compact.dropna(inplace=True) # drop any rows with missing values from the DataFrame
df_pitches_description = df_pitches_compact
#print(df_pitches_compact['description'].value_counts()) # verify that there are no longer any missing values in the DataFrame
if 'hit' not in df_pitches_description['description'].cat.categories:
    df_pitches_description['description'] = df_pitches_description['description'].cat.add_categories('hit')
# Group strike
if 'strike' not in df_pitches_description['description'].cat.categories:
    df_pitches_description['description'] = df_pitches_description['description'].cat.add_categories('strike')
if 'ball' not in df_pitches_description['description'].cat.categories:
    df_pitches_description['description'] = df_pitches_description['description'].cat.add_categories('hit')
# Group strike
df_pitches_description['description'] = df_pitches_description['description'].where(~df_pitches_description['description'].isin(['ball', 'blocked_ball', 'hit_by_pitch', 'pitchout']), 'ball')
df_pitches_description['description'] = df_pitches_description['description'].where(~df_pitches_description['description'].isin(['called_strike', 'swinging_strike', 'foul_tip', 'swinging_strike_blocked', 'foul_bunt', 'foul','missed_bunt', 'bunt_foul_tip']), 'strike')
df_pitches_description['description'] = df_pitches_description['description'].where(~df_pitches_description['description'].isin(['triple_play', 'sac_fly_double_play', 'catcher_interf', 'double_play', 'fielders_choice_out', 'fielders_choice', 'sac_bunt', 'field_error', 'sac_fly', 'grounded_into_double_play', 'force_out', 'field_out']), 'field_out')
df_pitches_description['description'] = df_pitches_description['description'].where(~df_pitches_description['description'].isin(['single', 'double', 'triple', 'home_run']), 'hit')

#removing unneeded categories
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['blocked_ball'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['called_strike'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['double'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['double_play'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['field_error'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['fielders_choice'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['fielders_choice_out'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['force_out'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['foul'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['foul_bunt'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['foul_tip'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['grounded_into_double_play'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['hit_by_pitch'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['home_run'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['missed_bunt'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['pitchout'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['sac_bunt'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['sac_fly'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['single'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['swinging_strike'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['swinging_strike_blocked'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['triple'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['bunt_foul_tip'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['triple_play'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['catcher_interf'])
df_pitches_description['description'] = df_pitches_description['description'].cat.remove_categories(['sac_fly_double_play'])

#printing value counts to make sure they appear accurate
print(df_pitches_description['description'].value_counts())
df_pitches_description.to_pickle("./cleanedPitchData.pkl")
 