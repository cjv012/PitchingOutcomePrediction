import numpy as np
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt
"""
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
"""
print('Hello World')
df_pitches = pd.read_csv('data/savant_data (2).csv')
for i in range(8):
    x = i+3
    print(x)
    df_new_rows = pd.read_csv('data/savant_data (' + str(x) + ').csv')
    df_pitches = pd.concat([df_pitches, df_new_rows])
# Set the index of the DataFrame to the 'player_name' column to allow for easier data access by player name
print(df_pitches.head(10))