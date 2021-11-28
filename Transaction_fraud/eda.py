"""Exploratory data analysis for credit card fraud data set.
Kaggle dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
The dataset contains credit cards transactions made by European cardholders
in two days in September 2013: a total of 492 frauds out of 284,807 transactions.
The positive class (frauds) account for 0.172% of all transactions.
Features:
V1, V2, … V28 - numerical features the principal components obtained with PCA
Time - the seconds elapsed between each transaction and the first transaction in the data set
Amount - the transaction amount
Target value:
Class - takes value 1 in case of fraud and 0 otherwise
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATH = '../input/creditcardfraud/creditcard.csv'


def set_display():
    """Function sets display options for charts and pd.DataFrames.
    """
    # Plots display settings
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = 12, 8
    plt.rcParams.update({'font.size': 14})
    # DataFrame display settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.options.display.float_format = '{:.4f}'.format


set_display()

data = pd.read_csv(PATH)
print(f'Data shape: {data.shape}')
print(data.head())

print(data.info())

print(data.describe())

# Target distribution: fraud transactions account for less than 1% of samples.
classes = data['Class'].value_counts(normalize=True)
plt.pie(classes.values, labels=classes.index, autopct='%1.1f%%')
plt.title('Class Imbalance')
plt.show()

# Transaction amount distribution
plt.hist(data['Amount'], bins=30, log=True)
plt.xlabel('Transaction amount')
plt.title('Amount Distribution')
plt.show()

# Time distribution of transactions
print(data['Time'].describe())

plt.hist(data['Time'], bins=30)
plt.xlabel('Seconds from transaction #1')
plt.title('Transactions Time')
plt.show()

# Feature engineering: temporal features
# Obtain hour from "Time" column, which represents
# seconds passed from the 1st transaction.
# Data set contains transactions for 2 days.
data['hour'] = data['Time'] // 3600
data['hour'] = data['hour'].apply(lambda x: x if x < 24 else x - 24)

hours = data['hour'].value_counts()
plt.bar(hours.index, hours.values)
plt.xlabel('Hour')
plt.ylabel('N transactions')
plt.title('Transactions by Hour')
plt.show()

# Distribution of numerical features
num_features = [col for col in data.columns if col.find('V') > -1]

n_cols = 2
n_rows = 14

fig = plt.gcf()
fig.set_size_inches(n_cols * 6, n_rows * 4)

for pos, feature in enumerate(num_features):
    sp = plt.subplot(n_rows, n_cols, pos + 1)
    plt.hist(data[feature], bins=30)
    plt.title(feature)

plt.show()

correlation = data.corr()
ax = sns.heatmap(correlation, center=0, cmap='RdBu_r')
l, r = ax.get_ylim()
ax.set_ylim(l + 0.5, r - 0.5)
plt.yticks(rotation=0)
plt.title('Correlation Matrix')
plt.show()

idx = [col for col in data.columns if col != 'Class']
plt.barh(correlation.loc[idx, 'Class'].index, correlation.loc[idx, 'Class'].values)
plt.title('Feature Correlation with Target')
plt.show()
