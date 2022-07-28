#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.model_selection import GridSearchCV

#Load data
url = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"
df = pd.read_csv(url)

#Transform data
def insulina(insulin_value, outcome_value, insuline_mean_0,insuline_mean_no0):
    if outcome_value==0 and insulin_value==0:
        return insuline_mean_0
    elif outcome_value==1 and insulin_value==0:
        return insuline_mean_no0
    else:
        return insulin_value

df['Insulin'] = df.apply(lambda x: insulina(x['Insulin'], x['Outcome'],insuline_mean_0,insuline_mean_no0), axis=1)

df_0=df[(df['Outcome']==0) & (df["Glucose"] > 0)]
glucose_mean_0=df_0['Glucose'].mean()

df_no0=df[(df['Outcome']!=0) & (df["Glucose"] > 0)]
glucose_mean_no0=df_no0['Glucose'].mean()

def glucose_fun(glucose_value, outcome_value, glucose_mean_0,glucose_mean_no0):
    if outcome_value==0 and glucose_value==0:
        return glucose_mean_0
    elif outcome_value==1 and glucose_value==0:
        return glucose_mean_no0
    else:
        return glucose_value

df['Glucose'] = df.apply(lambda x: glucose_fun(x['Glucose'], x['Outcome'],glucose_mean_0,glucose_mean_no0), axis=1)

df_0=df[(df['Outcome']==0) & (df["SkinThickness"] > 0)]
skin_mean_0=df_0['SkinThickness'].mean()

df_no0=df[(df['Outcome']!=0) & (df["SkinThickness"] > 0)]
skin_mean_no0=df_no0['SkinThickness'].mean()

def skin_fun(skin_value, outcome_value, skin_mean_0, skin_mean_no0):
    if outcome_value==0 and skin_value==0:
        return skin_mean_0
    elif outcome_value==1 and skin_value==0:
        return skin_mean_no0
    else:
        return skin_value

df['SkinThickness'] = df.apply(lambda x: skin_fun(x['SkinThickness'], x['Outcome'],skin_mean_0,skin_mean_no0), axis=1)

df_0=df[(df['Outcome']==0) & (df["BMI"] > 0)]
bmi_mean_0=df_0['BMI'].mean()

df_no0=df[(df['Outcome']!=0) & (df["BMI"] > 0)]
bmi_mean_no0=df_no0['BMI'].mean()

def bmi_fun(bmi_value, outcome_value, bmi_mean_0, bmi_mean_no0):
    if outcome_value==0 and bmi_value==0:
        return bmi_mean_0
    elif outcome_value==1 and bmi_value==0:
        return bmi_mean_no0
    else:
        return bmi_value

df['BMI'] = df.apply(lambda x: skin_fun(x['BMI'], x['Outcome'],bmi_mean_0,bmi_mean_no0), axis=1)

#Split data
X = df.drop('Outcome', axis = 1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 34)

#Decision tree with gini as criteria
clf_1 = DecisionTreeClassifier(random_state = 0) #Default criterion = gini
clf_1.fit(X_train, y_train)
clf_1.score(X_test, y_test)
clf_pred_1 = clf_1.predict(X_test)

#Decision tree with entropy as criteria
clf_2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
clf_2.fit(X_train, y_train)
clf_2.score(X_test, y_test)
clf_pred_2 = clf_2.predict(X_test)

#Grid search
tree_param = {'criterion':['gini','entropy'],'max_depth':[12,15,20],'min_samples_split': [2, 3, 4]}
clf_3 = GridSearchCV(DecisionTreeClassifier(), tree_param, cv=5)
clf_3.fit(X_train, y_train)
gsp = clf_3.best_estimator_
y_pred_gsp = gsp.predict(X_test)
