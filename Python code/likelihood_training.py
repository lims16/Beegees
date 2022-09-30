# version 5 of the Bayesian Classifier program
# Create likelihood tables, similar to pivot tables in Excel
import pandas as pd
import numpy as np
import random

df = pd.read_csv('training_data.csv')

# create empty array filled with zeros
# rows = rows of given data, 2x columns for x features + secure/not
likelihood = np.zeros(((df.shape[0]), (len(df.columns) - 1) * 2))
# rows = rows of given data, x columns for x features, excluding security feature
evidence = np.zeros(((df.shape[0]), len(df.columns) - 1))

# identify features
features = []
for c in df.columns:
    if c != "Secure":
        features.append(c)
print(features)

# create x pivot tables, x = number of features
for c in range(len(features)):
    # training part
    # counts secure
    df2 = df.groupby([features[c]]).Secure.value_counts(
    ).loc[:, 1]
    # counts not secure
    df3 = df.groupby([features[c]]).Secure.value_counts(
    ).loc[:, 0]
    # grand total col
    df4 = df.groupby([features[c]])['Secure'].count()
    df5 = [df3, df2, df4]
    result = pd.concat(df5, axis=1, sort="True")
    result = result.fillna(0)
    result.columns.values[0] = '0'
    result.columns.values[1] = '1'
    result.columns.values[2] = 'Grand Total'
    # the two lines below, when added to the code, replicate the Excel results which neglects NA in the pivot table
    if result.index.values[0] == ' ':
        result = result.drop(result.index[0])
    gtrow = result.sum(axis=0)
    result.loc[len(result.index)] = result.sum(axis=0)
    result.index.values[-1] = '-1'  # Grand Total
    pv = result
    # print(pv)
    # VLOOKUP - validation part
    vlup = pd.merge(df, pv, on=features[c], how='left')
    # print(vlup)
    print(pv)

    # store probabilities
    for r in range(df.shape[0]):
        # store data for P(xi) - evidence
        evidence[r, c] = vlup.iloc[r, -1] / pv.iloc[-1, -1]

        # store data for P(xi|Yes) - likelihood
        likelihood[r, c] = vlup.iloc[r, -2] / pv.iloc[-1, -2]

        # store data for P(xi|No) - likelihood
        likelihood[r, len(features) + c] = vlup.iloc[r, -3] / pv.iloc[-1, -3]

evidence = np.nan_to_num(evidence, nan=1)
likelihood = np.nan_to_num(likelihood, nan=1)

# P(secure) is the same for all features
securepv = df['Secure'].value_counts()
secure = securepv.loc[1] / (securepv.loc[1] + securepv.loc[0])
# print(secure)
