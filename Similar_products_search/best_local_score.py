"""Алгоритм для поиска товаров-аналогов на основе текстовых описаний и изображений.
Используется бинарная классификация: на основе сравнения параметров товаров
составляется список схожих пар, затем финальный классификатор определяет,
какие пары являются аналогами.
Извлечение параметров на основе анализа данных:
- Массивы, полученные путем преобразования текстовых описаний через TF-IDF
  и с помощью модели Universal sentence encoder, объединяются в общий массив векторов,
  на основе которого производится поиск потенциальных пар товаров-аналогов.
- Изображения преобразуются в массив векторов с использованием модели EfficientNetB7.
  На основе полученного массива производится поиск пар схожих товаров.
- Два списка потенциальных пар товаров объединяются и дополняются парами товаров,
  имеющих аналогичный phash, независимо от степени сходства текстов и изображений.
- Для каждой пары схожих товаров на основе текстовых описаний рассчитывается
  отношение количества уникальных слов, встречающихся в обоих текстах,
  к общему количеству уникальных слов в этих двух текстах.
- Добавляется бинарный параметр на основе phash для каждой потенциальной пары
  (1 означает идентичный phash, 0 - разный).
- Добавляется бинарный параметр на основе сравнения цифр в текстовых описаниях товаров
  для каждой потенциальной пары (1 означает одинаковые цифры, 0 - разные).
  Это позволяет выделить товары, отличающиеся, напрамер, по размеру или весу.
- Добавляются сводные числовые коэффициенты, показывающие степень схожести товаров
  и получаемые путем перемножения предыдущих коэффициентов.
Финальная модель - RandomForestClassifier - принимает в качестве входных данных
коэффициенты схожести и бинарные параметры для всех потенциальных пар товаров-аналогов.
"""

import numpy as np
import pandas as pd
import unicodedata
import string
import re
import queue
import math
import os
import gc
import itertools

import tensorflow as tf
from tensorflow_hub import KerasLayer
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns

# Plots display settings
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 14})

# TensorFlow settings
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = 299
BATCH_SIZE = 32

# Number of features (words in dict) for TF-IDF
tf_idf_features = 2000

# Train data paths
train_csv_path = '/kaggle/input/shopee-product-matching/train.csv'
train_img_path = '/kaggle/input/shopee-product-matching/train_images'

# Test data paths
test_csv_path = '/kaggle/input/shopee-product-matching/test.csv'
test_img_path = '/kaggle/input/shopee-product-matching/test_images'

# Universal sentence encoder model
txt_model_path = 'https://tfhub.dev/google/universal-sentence-encoder/4'

# Variable defines if the cross-validation will be performed for the final classifier
validation = True

# Target values and features to thain the classifier
target = ['true_match']

features = ['text_score', 'image_score', 'txt_img_score', 'words_ratio',
            'txt_img_words', 'phash_match', 'nums_match']

# Regex expression to remove traces of emoji from text
RE_SYMBOLS = re.compile("x\w\d\S+")

# ------------------------------- Functions -----------------------------------


def get_csv_data(csv_path: str, img_dir: str) -> pd.DataFrame:
    """Function reads data from a csv file, performs text cleaning
    in titles column and transforms image file names into file paths.
    :param csv_path: Path to a scv file
    :param img_dir: Path to directory with images
    :return Processed pd.DataFrame
    """
    data = pd.read_csv(csv_path)
    data['title'] = data['title'].apply(preprocess_titles)
    data['image'] = data['image'].apply(abs_path, args=(img_dir,))
    return data


def preprocess_titles(s: str) -> str:
    """Function converts text to lowercase, removes punctuation,
    replaces multiple spaces, normalizes and removes traces of emoji symbols.
    :param s: original text string
    :return: cleaned text string
    """
    s = RE_SYMBOLS.sub(r'', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub('\s+', ' ', s)
    s = s.lower()
    return unicodedata.normalize('NFKC', s)


def get_tfidf_features(n_features) -> np.array:
    """Function creates an array of TF-IDF features
    for titles referencing DataFrame objects from outer scope.
    :param n_features: Maximum number of words in TF-IDF dictionary
    :return: Array of TF-IDF features with shape n_items x n_features
    """
    # Transform all titles from the original DataFrame into TF-IDF matrix
    vectorizer = TfidfVectorizer(decode_error='ignore',
                                 stop_words='english',
                                 max_features=n_features)

    vectors = vectorizer.fit_transform(data['title']).toarray().astype(np.float16, copy=False)
    print('TF-IDF features extracted. Shape:', vectors.shape)

    return vectors


def get_text_features() -> np.array:
    """Function loads Universal sentence encoder model model from Kaggle dataset,
    transforms titles from the original DataFrame into feature matrix
    using embeddings from the pretrained language model.
    :returns: Features array with the shape n_samples x 512 features
    """
    # Universal sentence encoder model
    # Original model by Google could be loaded from: https://tfhub.dev/google/universal-sentence-encoder/4
    # In this notebook the model is loaded from a public dataset on Kaggle
    # at https://www.kaggle.com/dimitreoliveira/universalsentenceencodermodels
    text_model = tf.keras.Sequential(
        [KerasLayer(txt_model_path, input_shape=[], dtype=tf.string,  # Pretrained model
                    output_shape=[512], trainable=False),
         tf.keras.layers.Layer(512, dtype='float16')]  # This layer reduces precision of float numbers
    )

    # Convert all texts to vectors
    features = text_model.predict(data['title'],
                                  batch_size=BATCH_SIZE,
                                  use_multiprocessing=True,
                                  workers=-1)
    print('Text features extracted. Shape:', features.shape)

    return features


def get_image_features(paths: pd.Series) -> np.array:
    """Function loads pretrained image classification model from file,
    transforms images into feature matrix with the shape n_samples x n_features.
    :param paths: Series object containing paths to image files
    :returns: Features array with the shape n_samples x 2560 features
    """
    # Pretrained image classification model to convert images into embeddings
    image_model = tf.keras.applications.EfficientNetB7(weights='imagenet',
                                                       include_top=False,
                                                       input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                       pooling='avg')
    image_model = tf.keras.Sequential(
        [tf.keras.models.load_model(image_model),
         tf.keras.layers.Layer(2560, dtype='float16')]  # This layer reduces precision of float numbers
    )

    # Transform paths to files into tf.data.Dataset
    input_data = tf.data.Dataset.from_tensor_slices(paths)
    # Preprocess images
    input_data = input_data.map(process_path, num_parallel_calls=AUTOTUNE)
    input_data = configure_for_performance(input_data)

    # Convert all images into embeddings and average colors
    features = image_model.predict(input_data,
                                   batch_size=BATCH_SIZE,
                                   use_multiprocessing=True,
                                   workers=-1)
    print('Image features extracted. Shape:', features.shape)

    return features


@tf.function
def process_path(file_path: str):
    """Function reads image from the file and returns
    preprocessed image.
    :param file_path: Path to the image file
    :return Tensor with preprocessed image from the file
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.keras.applications.efficientnet.preprocess_input(img)  # Shape: IMG_SIZE x IMG_SIZE x 3


def configure_for_performance(ds):
    """Function applies batches and prefetches dataset
    to optimize data processing.
    :param ds: TensorFlow Dataset object
    :return Batched TensorFlow Dataset object with prefetch() applied
    """
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def word_intersection() -> list:
    """Function processes DataFrame with candidate item pairs
    and returns a list containing a ratio of intersecting words
    in title pairs to the union of unique words. References objects
    from outer scope.
    :return: List of float values ranging between 0 and 1.
    Length of list equals to the number of given item pairs.
    """

    def count_words(title_pair: np.array) -> float:
        """Function calculates number of unique words present
        in both titles and ratio of that number to the union of unique words.
        :param title_pair: Array containing two titles
        :return: Ratio of intersecting words to union of unique words
        """
        title_1, title_2 = title_pair
        # Transform into sets of words
        title_1 = set(title_1.split())
        title_2 = set(title_2.split())
        # Divide length of intersection by length of union
        ratio = len(title_1.intersection(title_2)) / len(title_1.union(title_2))
        return ratio

    # Find titles for each pair in the current chunk by their indexes
    tmp_df = pd.DataFrame()
    tmp_df['title_1'] = data.loc[pairs.loc[:, 'idx_1'].values, 'title'].values
    tmp_df['title_2'] = data.loc[pairs.loc[:, 'idx_2'].values, 'title'].values

    # Process title pairs in current chunk and add results to the list
    scores = [result for result in map(count_words, tmp_df[['title_1', 'title_2']].values)]

    return scores


def combinations(n) -> float:
    """Function calculates number of unique combinations
    of two titles among n samples with the same label group."""
    c = math.factorial(n) / (math.factorial(2) * math.factorial(n - 2))
    return c


def feature_mining(features: np.array,
                   query_chunk_size: int = 5000,
                   corpus_chunk_size: int = 100000,
                   max_pairs: int = 500000,
                   top_k: int = 100) -> list:
    """Given an array of features, this function performs data mining.
    It compares all items against all other items and returns a list
    with the pairs that have the highest cosine similarity score.

    :param features: np.array of shape n_samples * n_features for all items
    :param query_chunk_size: Search for most similar pairs for query_chunk_size at the same time.
           Decrease, to lower memory footprint (increases run-time).
    :param corpus_chunk_size: Compare an image simultaneously against corpus_chunk_size other items.
           Decrease, to lower memory footprint (increases run-time).
    :param max_pairs: Maximal number of item pairs returned.
    :param top_k: For each item, we retrieve up to top_k other items.
    :return: Returns a list of triplets with the format [score, id1, id2]
    """

    top_k += 1  # An image has the highest similarity to itself. Increase +1 as we are interest in distinct pairs

    # Mine for duplicates
    pairs = queue.PriorityQueue()
    min_score = -1
    num_added = 0

    for corpus_start_idx in range(0, len(features), corpus_chunk_size):
        corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(features))
        for query_start_idx in range(0, len(features), query_chunk_size):
            query_end_idx = min(query_start_idx + query_chunk_size, len(features))

            cos_scores = torch.Tensor(
                cosine_similarity(features[query_start_idx:query_end_idx],
                                  features[corpus_start_idx:corpus_end_idx])
            )

            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])),
                                                                       dim=1, largest=True, sorted=False)
            cos_scores_top_k_values = cos_scores_top_k_values.tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.tolist()

            for query_itr in range(len(cos_scores)):
                for top_k_idx, corpus_itr in enumerate(cos_scores_top_k_idx[query_itr]):
                    i = query_start_idx + query_itr
                    j = corpus_start_idx + corpus_itr

                    if i != j and cos_scores_top_k_values[query_itr][top_k_idx] > min_score:
                        pairs.put((cos_scores_top_k_values[query_itr][top_k_idx], i, j))
                        num_added += 1

                        if num_added >= max_pairs:
                            entry = pairs.get()
                            min_score = entry[0]

    # Get the pairs
    added_pairs = set()  # Used for duplicate detection
    pairs_list = []
    while not pairs.empty():
        score, i, j = pairs.get()
        sorted_i, sorted_j = sorted([i, j])

        if sorted_i != sorted_j and (sorted_i, sorted_j) not in added_pairs:
            added_pairs.add((sorted_i, sorted_j))
            pairs_list.append([score, i, j])

    return pairs_list


def check_identity(par: str) -> np.array:
    """Function finds values in 'par' column of the original DataFrame
    using 2 row indexes from DataFrame of candidate pairs and returns
    a binary column for match between two items (1 - matching, 0 - non-matching).
    :param par: parameter to check
    :return Series with binary values - result of the comparison
    """
    # Temporary dataframe to compare 'par' values for item pairs
    identity = pd.DataFrame()
    # Look up values by respective row indexes
    identity['item_1'] = data.iloc[pairs['idx_1'], :][par].values
    identity['item_2'] = data.iloc[pairs['idx_2'], :][par].values
    # Binary column signifying match or mismatch
    identity['match'] = (identity['item_1'] == identity['item_2']).astype('int')
    return identity['match'].values


def numbers_identity() -> np.array:
    """Function extracts numbers from title pairs and compares them,
    returns a binary column for match between numbers in two titles.
    :return Series with binary values - result of the comparison
    """
    # Temporary dataframe to compare numbers for item pairs
    identity = pd.DataFrame()
    # Look up values by respective row indexes
    identity['item_1'] = data.iloc[pairs['idx_1'], :]['title'].values
    identity['item_2'] = data.iloc[pairs['idx_2'], :]['title'].values
    # Extract numbers and convert them to space-delimited string
    identity['nums_1'] = identity['item_1'].apply(lambda x: ' '.join(re.findall(r'\d+', x)))
    identity['nums_2'] = identity['item_2'].apply(lambda x: ' '.join(re.findall(r'\d+', x)))
    # Binary column signifying match or mismatch
    identity['match'] = (identity['nums_1'] == identity['nums_2']).astype('int')
    return identity['match'].values


def check_pairs(reference_df: pd.DataFrame, check_labels=False) -> list:
    """Function finds indexes of all phash pairs and label group pairs
    (for train set only) in the original DataFrame.
    :param reference_df: original DataFrame with items
    :param check_labels: boolean flag, if True - add indexes for all known title pairs
           from the train set regardless of similarity scores and phash values
    :return: Returns a list of index pairs
    """
    print('Number of candidate pairs:', len(pairs))
    # Indexes of add pairs with identical phash
    all_pairs = set()
    groups = reference_df.groupby(by='image_phash')
    # Numeric index of rows in train set as a temporary column
    reference_df['num_idx'] = [idx for idx in range(len(reference_df))]
    for group in groups.indices:
        num_index = groups.get_group(group)['num_idx']
        # All combinations of index pairs with the same phash
        cur_pairs = set(itertools.combinations(num_index, 2))
        all_pairs = all_pairs.union(cur_pairs)
    print(f'Total number of phash pairs: {len(all_pairs)}')

    # When dealing with train set, add indexes of all label group pairs
    if check_labels:
        groups = reference_df.groupby(by='label_group')
        for group in groups.indices:
            num_index = groups.get_group(group)['num_idx']
            # All combinations of index pairs with the same label group
            cur_pairs = set(itertools.combinations(num_index, 2))
            all_pairs = all_pairs.union(cur_pairs)
        print(f'With labels added, total number of pairs: {len(all_pairs)}')

    return list(all_pairs)


def abs_path(file_name: str, directory: str) -> str:
    """Function returns a Series of absolute paths to images
    given file names and directory name.
    :param file_name: Name of the image file
    :param directory: Name of directory containing the file
    :return Path to the image file
    """
    return os.path.join(directory, file_name)


def analyze_similarities():
    """Function compares number of actual item pairs
    for various cosine similarity thresholds and binary features.
    """
    print('Total number of candidate pairs:', len(pairs))
    print(f'\nNumber of actual item pairs in the train set: {pairs["true_match"].sum()}\n')

    for feature in ['text_score', 'image_score', 'txt_img_score', 'words_ratio', 'txt_img_words']:

        # Check distribution of True and False predictions for various similarity scores
        print('-' * 50)
        print(f'\nDistribution of True/False predictions for {feature}')
        for thr in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
            print('-' * 50)
            print(f'Similarity score over {thr}')
            pairs_sample = pairs[pairs[feature] >= thr]
            print(f'Number of similar item pairs: {len(pairs_sample)}')
            print(pairs_sample['true_match'].value_counts(normalize=True))

    # Check if identical phash can be used to improve the accuracy
    same_phash = pairs[pairs['phash_match'] == 1]
    different_phash = pairs[pairs['phash_match'] == 0]

    print('\nFor item pairs with the same phash:')
    print(same_phash['true_match'].value_counts(normalize=True))
    print('Number of item pairs in this subset:', len(same_phash))

    print('\nFor item pairs with different phash:')
    print(different_phash['true_match'].value_counts(normalize=True))
    print('Number of item pairs in this subset:', len(different_phash))

    # Check if numbers in titles can be used to improve the accuracy
    same_numbers = pairs[pairs['nums_match'] == 1]
    different_numbers = pairs[pairs['nums_match'] == 0]

    print('\nFor item pairs with the same numbers:')
    print(same_numbers['true_match'].value_counts(normalize=True))
    print('Number of item pairs in this subset:', len(same_numbers))

    print('\nFor item pairs with different numbers:')
    print(different_numbers['true_match'].value_counts(normalize=True))
    print('Number of item pairs in this subset:', len(different_numbers))


def correlation_and_distribution(pairs: pd.DataFrame, features: list, target: list):
    """Function plots a heatmap of correlation for features
    in 'pairs' and distribution of feature values for two classes.
    :param pairs: DataFrame with candidate item pairs
    :param features: List of input features names
    :param target: List with target name
    """
    # Correlation between all features and predicted binary target
    ax = sns.heatmap(pairs[features + target].corr(),
                     center=0, annot=True, cmap='RdBu_r')
    l, r = ax.get_ylim()
    ax.set_ylim(l + 0.5, r - 0.5)
    plt.yticks(rotation=0)
    plt.title('Correlation matrix')
    plt.show()

    # Distribution of feature values for two classes
    similar = pairs[pairs['true_match'] == 1]
    different = pairs[pairs['true_match'] == 0]

    for feature in features:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(similar[feature], bins=30)
        plt.title(f'{feature} in similar samples')
        plt.subplot(1, 2, 2)
        plt.hist(different[feature], bins=30)
        plt.title(f'{feature} in different samples')
        plt.show()


def find_similar_items(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    """Function returns pd.Series with space-delimited IDs
    for products with similar titles, including self-match.
    :param df: DataFrame with original data
    :param pairs: DataFrame with candidate pairs
    :return Series object with space-delimited IDs
    """

    # Add columns with posting IDs from the original DataFrame
    pairs['id_1'] = pairs['idx_1'].apply(lambda x: df.loc[x, 'posting_id'])
    pairs['id_2'] = pairs['idx_2'].apply(lambda x: df.loc[x, 'posting_id'])

    # Group posting IDs by id_1
    id_1 = pd.DataFrame(pairs.groupby('id_1')['id_2'].unique())

    # Convert lists to space-delimited strings
    id_1['id_2'] = id_1['id_2'].apply(lambda x: ' '.join(x))
    # Convert to dictionary
    id_1 = id_1.to_dict()['id_2']
    # Create a column for self-match and other matching IDs (space-delimited)
    df['title_match'] = df['posting_id'] + ' '
    df['title_match'] = df['title_match'] + df['posting_id'].apply(lambda x: id_1[x] if x in id_1 else '')

    # Group posting IDs by id_2
    id_2 = pd.DataFrame(pairs.groupby('id_2')['id_1'].unique())
    id_2['id_1'] = id_2['id_1'].apply(lambda x: ' '.join(x))
    id_2 = id_2.to_dict()['id_1']
    df['title_match'] = df['title_match'] + ' ' + df['posting_id'].apply(lambda x: id_2[x] if x in id_2 else '')

    return df['title_match']


# ---------------------------- Processing train set ---------------------------

# Load train data
data = get_csv_data(csv_path=train_csv_path, img_dir=train_img_path)

# Get TF-IDF vectors
features_arr = get_tfidf_features(n_features=tf_idf_features)
gc.collect()  # Launch garbage collector to avoid memory errors

# Get text embeddings
features_arr = np.hstack((features_arr, get_text_features()))
gc.collect()

# Get a list of similar item pairs based on text
pairs = feature_mining(features_arr, query_chunk_size=1000, corpus_chunk_size=30000, top_k=100)
pairs = pd.DataFrame(pairs, columns=['text_score', 'idx_1', 'idx_2'])
gc.collect()

# Order index pairs so that first index is lower than the second one
# to avoid duplicates when adding phash and label pairs
ordered = pairs[['idx_1', 'idx_2']].agg(['min', 'max'], axis='columns')
pairs[['idx_1', 'idx_2']] = ordered.values

# Get image embeddings
img_features_arr = get_image_features(data['image'])
gc.collect()

# Get a list of similar item pairs based on images
img_pairs = feature_mining(img_features_arr, query_chunk_size=1000, corpus_chunk_size=30000, top_k=100)
img_pairs = pd.DataFrame(img_pairs, columns=['image_score', 'idx_1', 'idx_2'])
gc.collect()

# Order index pairs so that first index is lower than the second one
# to avoid duplicates when adding phash and label pairs
ordered = img_pairs[['idx_1', 'idx_2']].agg(['min', 'max'], axis='columns')
img_pairs[['idx_1', 'idx_2']] = ordered.values

# Combine all candidate pairs
pairs = pd.merge(pairs, img_pairs, how='outer')
del img_pairs
gc.collect()

# Get indexes for all pairs with identical phash and label
all_pairs = pd.DataFrame(check_pairs(data, check_labels=True), columns=['idx_1', 'idx_2'])
pairs = pairs.append(all_pairs, ignore_index=True).drop_duplicates(subset=['idx_1', 'idx_2'])
gc.collect()

# Fill in missing values in similarity scores in chunks of size 5,000
chunk = 5_000
while True:
    if pairs['text_score'].isna().sum() == 0:
        break
    else:
        replace_idx = pairs[pairs['text_score'].isna()].head(chunk).index
        # Indexes for image pairs with missing image similarity scores
        idx_1 = pairs[pairs['text_score'].isna()].head(chunk)['idx_1'].values
        idx_2 = pairs[pairs['text_score'].isna()].head(chunk)['idx_2'].values
        # Pass both feature arrays through Dot layer to get cosine similarity scores
        pairs.loc[replace_idx, 'text_score'] = np.diagonal(
            cosine_similarity(features_arr[idx_1], features_arr[idx_2])
        )
del features_arr
gc.collect()
while True:
    if pairs['image_score'].isna().sum() == 0:
        break
    else:
        replace_idx = pairs[pairs['image_score'].isna()].head(chunk).index
        # Indexes for image pairs with missing image similarity scores
        idx_1 = pairs[pairs['image_score'].isna()].head(chunk)['idx_1'].values
        idx_2 = pairs[pairs['image_score'].isna()].head(chunk)['idx_2'].values
        # Pass both feature arrays through Dot layer to get cosine similarity scores
        pairs.loc[replace_idx, 'image_score'] = np.diagonal(
            cosine_similarity(img_features_arr[idx_1], img_features_arr[idx_2])
        )
del img_features_arr
gc.collect()

# Multuplicative scores
pairs['txt_img_score'] = pairs['text_score'] * pairs['image_score']

# Ratio of intersecting words in title pairs to union of unique words
pairs['words_ratio'] = word_intersection()
gc.collect()

# Multiplicative feature
pairs['txt_img_words'] = pairs['txt_img_score'] * pairs['words_ratio']

# Add binary features (0 or 1):
# Ground truth labels for training
pairs['true_match'] = check_identity('label_group')

# Same phash
pairs['phash_match'] = check_identity('image_phash')

# Identical numbers in titles
pairs['nums_match'] = numbers_identity()
gc.collect()

print(pairs.head())

# Visualize correlation between features and target
# and distribution of values in two classes
correlation_and_distribution(pairs, features, target)

# Final classifier
clf = RandomForestClassifier(min_samples_leaf=10, n_jobs=-1)

# If we run this notebook for cross-validation
if validation:
    skf = StratifiedKFold(3, shuffle=True, random_state=1)
    scores = cross_val_score(clf, pairs[features], pairs[target].values.ravel(), cv=skf, scoring='f1')
    print('Cross-validation scores:', scores)
    print('Average train CV score:', np.mean(scores))

# Train the ensemble using all available data
clf.fit(
    pairs[features],
    pairs[target].values.ravel()
)

# ------------------------- Applying models to test set -------------------------

# Load test data
data = get_csv_data(csv_path=test_csv_path, img_dir=test_img_path)

# To avoid errors when running this code with only limited access to test set
if len(data) == 3:
    data['matches'] = data['posting_id']
    data[['posting_id', 'matches']].to_csv('/kaggle/working/submission.csv', index=False)

# When running this code for actual submission (70,000 samples)
else:
    # Get TF-IDF vectors
    features_arr = get_tfidf_features(n_features=tf_idf_features)
    gc.collect()  # Launch garbage collector to avoid memory errors

    # Get text embeddings
    features_arr = np.hstack((features_arr, get_text_features()))
    gc.collect()

    # Get a list of similar item pairs based on text
    pairs = feature_mining(features_arr, query_chunk_size=1000, corpus_chunk_size=20000, top_k=100)
    pairs = pd.DataFrame(pairs, columns=['text_score', 'idx_1', 'idx_2'])
    gc.collect()

    # Order index pairs so that first index is lower than the second one
    # to avoid duplicates when adding phash and label pairs
    ordered = pairs[['idx_1', 'idx_2']].agg(['min', 'max'], axis='columns')
    pairs[['idx_1', 'idx_2']] = ordered.values

    # Get image embeddings
    img_features_arr = get_image_features(data['image'])
    gc.collect()

    # Get a list of similar item pairs based on images
    img_pairs = feature_mining(img_features_arr, query_chunk_size=1000, corpus_chunk_size=20000, top_k=100)
    img_pairs = pd.DataFrame(img_pairs, columns=['image_score', 'idx_1', 'idx_2'])
    gc.collect()

    # Order index pairs so that first index is lower than the second one
    # to avoid duplicates when adding phash and label pairs
    ordered = img_pairs[['idx_1', 'idx_2']].agg(['min', 'max'], axis='columns')
    img_pairs[['idx_1', 'idx_2']] = ordered.values

    # Combine all candidate pairs
    pairs = pd.merge(pairs, img_pairs, how='outer')
    del img_pairs
    gc.collect()

    # Get indexes for all pairs with identical phash and label
    all_pairs = pd.DataFrame(check_pairs(data, check_labels=True), columns=['idx_1', 'idx_2'])
    pairs = pairs.append(all_pairs, ignore_index=True).drop_duplicates(subset=['idx_1', 'idx_2'])
    gc.collect()

    # Fill in missing values in similarity scores in chunks of size 5,000
    chunk = 5_000
    while True:
        if pairs['text_score'].isna().sum() == 0:
            break
        else:
            replace_idx = pairs[pairs['text_score'].isna()].head(chunk).index
            # Indexes for image pairs with missing image similarity scores
            idx_1 = pairs[pairs['text_score'].isna()].head(chunk)['idx_1'].values
            idx_2 = pairs[pairs['text_score'].isna()].head(chunk)['idx_2'].values
            # Pass both feature arrays through Dot layer to get cosine similarity scores
            pairs.loc[replace_idx, 'text_score'] = np.diagonal(
                cosine_similarity(features_arr[idx_1], features_arr[idx_2])
            )
    del features_arr
    gc.collect()
    while True:
        if pairs['image_score'].isna().sum() == 0:
            break
        else:
            replace_idx = pairs[pairs['image_score'].isna()].head(chunk).index
            # Indexes for image pairs with missing image similarity scores
            idx_1 = pairs[pairs['image_score'].isna()].head(chunk)['idx_1'].values
            idx_2 = pairs[pairs['image_score'].isna()].head(chunk)['idx_2'].values
            # Pass both feature arrays through Dot layer to get cosine similarity scores
            pairs.loc[replace_idx, 'image_score'] = np.diagonal(
                cosine_similarity(img_features_arr[idx_1], img_features_arr[idx_2])
            )
    del img_features_arr
    gc.collect()

    # Multuplicative scores
    pairs['txt_img_score'] = pairs['text_score'] * pairs['image_score']

    # Ratio of intersecting words in title pairs to union of unique words
    pairs['words_ratio'] = word_intersection()
    gc.collect()

    # Multiplicative feature
    pairs['txt_img_words'] = pairs['txt_img_score'] * pairs['words_ratio']

    # Add binary features
    pairs['phash_match'] = check_identity('image_phash')
    pairs['nums_match'] = numbers_identity()
    gc.collect()
    print(pairs.head())

    # Predict matching pairs with classification model
    pairs['pred_match'] = clf.predict(pairs[features])

    # Drop all pairs that did not pass the final classifier
    pairs = pairs[pairs['pred_match'] == 1]

    # Add a column with space-delimited IDs for matching items to the test data set
    data['matches'] = find_similar_items(data, pairs)

    # Save two required columns to csv file
    data[['posting_id', 'matches']].to_csv('/kaggle/working/submission.csv', index=False)
