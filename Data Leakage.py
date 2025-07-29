""" Data leakage (or leakage) happens when your 
 training data contains information about the target


Target leakage occurs when your predictors include data that will 
not be available at the time you make predictions


If your validation is based on a simple train-test split,exclude the 
validation data from any type of fitting, including the fitting of preprocessing steps.
"""

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


data = pd.read_csv('AER_credit_card_data.csv', 
                   true_values = ['yes'], false_values = ['no'])

y = data.card

X = data.drop(['card'], axis=1)

print("Number of rows in the dataset:", X.shape[0])
print(X.head())

my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
cv_scores = cross_val_score(my_pipeline, X, y, 
                            cv=5,
                            scoring='accuracy')

print("Cross-validation accuracy: %f" % cv_scores.mean())

expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]


print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))
print('Fraction of those who received a card and had no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))


