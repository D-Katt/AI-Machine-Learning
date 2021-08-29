"""EDA and baseline model for scoring text readability based on simple text features.
VotingRegressor is used as a final classifier on top of 3 regression models.
"""

import string
import re
from collections import Counter

import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor, 
    GradientBoostingRegressor, VotingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge, SGDRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
import seaborn as sns

# Plots display settings
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 14})

# Paths to original data
train_path = '../input/commonlitreadabilityprize/train.csv'
test_path = '../input/commonlitreadabilityprize/test.csv'

# Text features for the regression model
features = ['n_chars', 'n_words', 'n_sent', 'punct_signs', 'quotes',
            'quotes_per_sent', 'signs_per_sent', 'words_per_sent',
            'chars_per_sent', 'chars_per_word', 'n_vowels', 'n_digits',
            'vowels_per_word', 'vowels_ratio', 'n_unique', 'unique_ratio']

target = 'target'

# Models to compare
models = [('Random Forest', RandomForestRegressor()),
          ('Ada Boost', AdaBoostRegressor()),
          ('Gradient Boosting', make_pipeline(StandardScaler(), GradientBoostingRegressor())),
          ('KNN', make_pipeline(StandardScaler(), KNeighborsRegressor())),
          ('Bayesian Ridge', make_pipeline(StandardScaler(), BayesianRidge())),
          ('SGD', make_pipeline(StandardScaler(), SGDRegressor()))]

# ---------------------------------- Functions --------------------------------------


def get_data(path: str) -> pd.DataFrame:
    """Function loads data from the csv file
    and creates features based on texts.
    :param path: Path to csv file with original data
    :return: DataFrame with original data and generated features
    """
    df = pd.read_csv(path)

    # Cleaned text (lowercase, punctuation removed)
    df['text'] = df['excerpt'].apply(preprocess_text)

    # Features based on text excerpts
    df['n_chars'] = df['text'].apply(len)  # Total number of characters without punctuation
    df['n_words'] = df['text'].apply(lambda s: len(s.split(' ')))  # Total number of words
    # Number of sentences, punctuation signs, quotes and vowels
    tmp = df['excerpt'].apply(process_characters)
    df[['n_sent', 'punct_signs', 'quotes', 'n_vowels', 'n_digits']] = tmp.apply(pd.Series)
    df['quotes_per_sent'] = df['quotes'] / df['n_sent']  # Average number of quotes per sentence
    df['signs_per_sent'] = df['punct_signs'] / df['n_sent']  # Average number of signs per sentence
    df['words_per_sent'] = df['n_words'] / df['n_sent']  # Average number of words per sentence
    df['chars_per_sent'] = df['n_chars'] / df['n_sent']  # Average number of characters per sentence
    df['chars_per_word'] = df['n_chars'] / df['n_words']
    df['vowels_per_word'] = df['n_vowels'] / df['n_words']  # Average number of vowels per word
    df['vowels_ratio'] = df['n_vowels'] / df['n_chars']  # Number of vowels to the total number of characters
    df['n_unique'] = df['text'].apply(lambda x: set(x).__len__())  # Number of unique words
    df['unique_ratio'] = df['n_unique'] / df['n_words']  # Number of unique words to total n_words

    print(df.isna().sum())

    return df


def preprocess_text(s: str) -> str:
    """Function converts text to lowercase, removes punctuation
    and replaces multiple spaces.
    :param s: original text string
    :return: cleaned text string
    """
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub('\s+', ' ', s)
    return s.lower()


def process_characters(s: str) -> tuple:
    """Function calculates total number of punctuation signs in the text,
    number of sentences assuming any sentence contains just one '.' or '?' or '!'
    and the number of quotes in the text.
    :param s: Original text string with punctuation
    :return: Tuple with 5 int values: total number of sentences,
    total number of punctuation signs, number of quotation pairs,
    number of vowels, number of digits
    """
    chars = Counter(s)
    # Number of sentences
    sentences = 0
    for char in '.?!':
        sentences += chars[char]
    # Total number of punctuation signs
    punct_signs = 0
    for char in string.punctuation:
        punct_signs += chars[char]
    # Number of vowels
    vowels = 0
    for char in 'aeiouy':
        vowels += chars[char]
    # Number of digits
    digits = 0
    for char in '0123456789':
        digits += chars[char]
    return sentences, punct_signs, chars['"'] // 2, vowels, digits


def correlation_distribution(df: pd.DataFrame):
    """Function visualizes correlation between features
    and target value for the train set.
    :param df: DataFrame with original data and generated features
    """
    # Correlation heatmap
    ax = sns.heatmap(df.corr(),
                     center=0, annot=True, cmap='RdBu_r')
    l, r = ax.get_ylim()
    ax.set_ylim(l + 0.5, r - 0.5)
    plt.yticks(rotation=0)
    plt.title('Correlation matrix')
    plt.show()

    # Individual correlations between features and the target
    for feature in features:
        print(f'{feature}: {df[feature].min()} - {df[feature].max()}, '
              f'mean = {df[feature].mean()}, median = {df[feature].median()}')
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(df[feature], df[target])
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.title(f'{feature} correlation')
        plt.subplot(1, 2, 2)
        plt.hist(df[feature], bins=20)
        plt.title(f'{feature} distribution')
        plt.show()

    # Distribution of target values and standard errors
    for par in ('target', 'standard_error'):
        print(f'{par}: {df[par].min()} - {df[par].max()}, mean = {df[par].mean()}, median = {df[par].median()}')
        plt.hist(df[par], bins=20)
        plt.title(f'{par} distribution')
        plt.show()


# --------------------------- Processing training data ------------------------------

# Load original data and generate text features
data = get_data(train_path)
print(data.isna().sum())

# Correlation between all features and target
correlation_distribution(data)

# Cross-validation of base models
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for name, model in models:
    scores = cross_val_score(model,
                             data[features],
                             data[target],
                             cv=kf,
                             scoring='neg_root_mean_squared_error',
                             n_jobs=-1)
    print(f'{name} RMSE = {scores.mean()}\tScores: {scores}')

# Voting of best regressors
voting_models = [('Gradient Boosting', make_pipeline(StandardScaler(), GradientBoostingRegressor())),
                 ('Bayesian Ridge', make_pipeline(StandardScaler(), BayesianRidge())),
                 ('SGD', make_pipeline(StandardScaler(), SGDRegressor()))]

clf = VotingRegressor(voting_models)

scores = cross_val_score(clf,
                         data[features],
                         data[target],
                         cv=kf,
                         scoring='neg_root_mean_squared_error',
                         n_jobs=-1)

print(f'Voting regressor RMSE = {scores.mean()}\tScores: {scores}')

# Train the classifier on the whole data set
clf.fit(data[features], data[target])

# ---------------------------- Processing test data --------------------------------

# Load original data and generate text features
data = get_data(test_path)

# Predicted scores
data['target'] = clf.predict(data[features])
print(data[['id', 'target']])

# Save the result
data[['id', 'target']].to_csv('submission.csv', index=False)
