# Predicting-Breast-Cancer-through-ML
Predicting Breast Cancer in Patients using Ensemble Techniques and Optimized Support Vector Classifier
Problem Statement : Given the details of cell nuclei taken from breast mass, predict whether or not a patient has breast cancer using the Ensembling Techniques. Perform necessary exploratory data analysis before building the model and evaluate the model based on performance metrics other than model accuracy

Importing the packages

import numpy as np import pandas as pd import matplotlib.pyplot as plt import seaborn as sns import plotly.express as px

from sklearn.svm import SVC from sklearn.ensemble import RandomForestClassifier from sklearn.linear_model import LogisticRegression from sklearn.model_selection import train_test_split,GridSearchCV from sklearn.metrics import accuracy_score,classification_report,confusion_matrix from sklearn.preprocessing import StandardScaler
