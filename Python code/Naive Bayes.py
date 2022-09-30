# version 5 of the Bayesian Classifier program
# to obtain accuracy for training data, replace:
# from likelihood_validation to likelihood_training
# from validation_data.csv to training_data.csv
import numpy as np
import pandas as pd
from likelihood_validation import secure, likelihood, evidence

df = pd.read_csv('validation_data.csv')

posterior_yes = np.zeros(((df.shape[0]), 1))
posterior_no = np.zeros(((df.shape[0]), 1))
security = np.zeros(((df.shape[0]), 1))
validation = np.zeros(((df.shape[0]), 1))
FP = 0
FN = 0

# for normalization
nposterior_yes = np.zeros(((df.shape[0]), 1))
nposterior_no = np.zeros(((df.shape[0]), 1))


for r in range(df.shape[0]):
    # P(Yes|xi) and P(xi|No) - posterior probabilities
    posterior_yes[r, 0] = secure * \
        np.prod(likelihood[r, :len(df.columns) - 1]) / np.prod(evidence[r, :])
    posterior_no[r, 0] = (1 - secure) * \
        np.prod(likelihood[r, len(df.columns) - 1:]) / np.prod(evidence[r, :])

    nposterior_yes[r, 0] = posterior_yes[r, 0] / \
        (posterior_yes[r, 0] + posterior_no[r, 0])
    nposterior_no[r, 0] = posterior_no[r, 0] / \
        (posterior_yes[r, 0] + posterior_no[r, 0])

    # round up to 10 decimal places
    nposterior_yes[r, 0] = round(nposterior_yes[r, 0], 10)
    nposterior_no[r, 0] = round(nposterior_no[r, 0], 10)

    # validate
    # Secure = 1, Not secure = 0
    if nposterior_yes[r, 0] > nposterior_no[r, 0]:
        security[r, 0] = 1
    else:
        security[r, 0] = 0

    # TRUE - 1, FALSE - 0
    if security[r, 0] == df.iloc[r, -1]:
        validation[r, 0] = 1
    else:
        validation[r, 0] = 0
        if df.loc[r, "Secure"] == 1:
            FN += 1
        else:
            FP += 1
# print(posterior_yes)
frequency = np.unique(validation, return_counts=True)
# print(frequency)
if len(frequency[0]) == 1:
    accuracy = 100
else:
    accuracy = frequency[1][1] / (frequency[1][1] + frequency[1][0]) * 100
print("The accuracy is ", accuracy)

# TYPE I AND II ERRORS
AP = df['Secure'].value_counts()[1]
AN = df['Secure'].value_counts()[0]
TYP1 = FP / AN * 100
TYP2 = FN / AP * 100
print("Type I error is ", TYP1)
print("Type II error is ", TYP2)
