"""Алгоритм для поиска товаров-аналогов на основе текстовых описаний и изображений.
Используется бинарная классификация: на основе сравнения параметров товаров
составляется список схожих пар, затем ансамбль моделей классификации определяет,
какие пары являются аналогами.
Извлечение параметров на основе анализа данных:
- С помощью готовой модели EfficientNetB7 из keras.applications с финальным слоем global average pooling
  все имеющиеся изображения преобразуются в 1D векторы, содержащие по 2560 параметров.
- С помощью модели Universal sentence encoder все текстовые описания товаров преобразуются в 1D векторы,
  содержащие 512 параметров.
- Полученные массивы используются для поиска наиболее схожих пар товаров -
  отдельно на основе текстов и отдельно по изображениям.
  Для этого применяется заимствованная из библиотеки sentence_transformers функция,
  модифицированная для данной задачи и позволяющая осуществлять перебор
  в рамках отдельных сегментов массива, что экономит память.
- Списки наиболее схожих текстов и изображений (для каждой пары имеются индексы
  и cosine similarity score) объединяются. Такой способ поиска схожих товаров связан с тем,
  что объекты-аналоги могут быть похожи только по описанию, но иметь разные иллюстрации и наоборот.
- К списку схожих пар товаров добавляются все оставшиеся пары с одинаковым phash (perceptual hash) -
  вне зависимости от степени схожести текстов и изображений.
- Образовавшиеся при объединении пропуски заполняются путем обращения к исходным массивам параметров
  для текстов и изображений и рассчета для них соответсвующего similarity score.
- Для всех найденных потенциальных пар (около 550 000 пар товаров) дополнительно рассчитывается
  cosine similarity score на основе сравнения TF-IDF векторов текстов.
- Для каждой пары рассчитывается относительная разность средних цветов по 3 цветовым каналам:
  abs(R1 - R2) / 255, abs(G1 - G2) / 255, abs(B1 - B2) / 255 и средняя разность по всем цветам.
  Это позволяет идентифицировать пары изображений, где показаны аналогичные товары разных цветов.
  В контексте поставленной задачи такие товары часто относятся к разным группам.
- Добавляется бинарный параметр на основе phash для каждой потенциальной пары
  (1 означает идентичный phash, 0 - разный).
- Добавляется бинарный параметр на основе сравнения цифр в текстовых описаниях товаров
  для каждой потенциальной пары (1 означает одинаковые цифры, 0 - разные).
  Это позволяет выделить товары, отличающиеся, напрамер, по размеру или весу.
- Добавляются сводные числовые коэффициенты, показывающие степень схожести товаров
  и получаемые путем перемножения предыдущих коэффициентов.
Финальная модель - VotingClassifier - объединяет модели GaussianNB, RandomForestClassifier
и KNeighborsClassifier и принимает в качестве входных данных коэффициенты схожести
и бинарные параметры для всех потенциальных пар товаров-аналогов.
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
from torch import Tensor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

# Plots display settings
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 14})

# TensorFlow settings
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = 299
BATCH_SIZE = 32

# Train data paths
train_csv_path = '/kaggle/input/shopee-product-matching/train.csv'
train_img_path = '/kaggle/input/shopee-product-matching/train_images'

# Test data paths
test_csv_path = '/kaggle/input/shopee-product-matching/test.csv'
test_img_path = '/kaggle/input/shopee-product-matching/test_images'

# Universal sentence encoder model
txt_model_path = 'https://tfhub.dev/google/universal-sentence-encoder/4'

# Variable to use for cross-validating final classifier.
# In submission notebook it is set to False.
scoring = False

# Regex expression to remove traces of emoji from text
RE_SYMBOLS = re.compile("x\w\d\S+")

# ------------------------------- Functions -------------------------------------


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


def combinations(n) -> float:
    """Function calculates number of unique combinations
    of two titles among n samples with the same label group."""
    c = math.factorial(n) / (math.factorial(2) * math.factorial(n - 2))
    return c


def get_image_features(paths: pd.Series) -> tuple:
    """Function loads pretrained image classification model from file,
    transforms images into feature matrix with the shape n_samples x n_features
    and finds most similar image pairs based on cosine similarity.
    :param paths: Series object containing paths to image files
    :returns: Tuple where 0-th element is the feature matrix, 1-st element is pairs DataFrame
     and 2-nd element is the minimum similarity score for selected pairs
    """
    # Pretrained image classification model EfficientNetB7
    image_model = tf.keras.applications.EfficientNetB7(weights='imagenet',
                                                       include_top=False,
                                                       input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                       pooling='avg')
    input_data = tf.data.Dataset.from_tensor_slices(paths)
    # Preprocess images
    input_data = input_data.map(process_path, num_parallel_calls=AUTOTUNE)
    input_data = configure_for_performance(input_data)
    # Features for all images
    image_features = image_model.predict(input_data, use_multiprocessing=True, workers=-1)
    print('Image features extracted. Shape:', image_features.shape)
    # Find similar image pairs
    pairs = feature_mining(image_features, query_chunk_size=1000, top_k=100)
    pairs = pd.DataFrame(pairs, columns=['image_score', 'idx_1', 'idx_2'])
    # Decrease precision for better memory usage
    image_features = image_features.astype(np.float16, copy=False)
    # Order index pairs so that first index is lower than the second one
    # to avoid duplicates when merging text pairs with image pairs
    ordered = pairs[['idx_1', 'idx_2']].agg(['min', 'max'], axis='columns')
    pairs[['idx_1', 'idx_2']] = ordered.values
    print(f'Number of image pairs: {len(pairs)}')
    # Minimum similarity score among image pairs returned by the image model
    image_sim_threshold = pairs['image_score'].min()
    print('Minimum image similarity score:', image_sim_threshold)
    return image_features, pairs, image_sim_threshold


def get_text_features(texts: pd.Series) -> tuple:
    """Function loads Universal sentence encoder model model from Kaggle dataset,
    transforms titles into feature matrix with the shape n_samples x 512 features
    and finds most similar title pairs based on cosine similarity.
    :param texts: Series object containing item titles
    :returns: Tuple where 0-th element is the feature matrix, 1-st element is pairs DataFrame
     and 2-nd element is the minimum similarity score among all selected pairs
    """
    # Universal sentence encoder model
    # Original model by Google could be loaded from: https://tfhub.dev/google/universal-sentence-encoder/4
    # In this notebook the model is loaded from a public dataset on Kaggle
    # at https://www.kaggle.com/dimitreoliveira/universalsentenceencodermodels
    text_model = tf.keras.Sequential(
        [KerasLayer(txt_model_path, input_shape=[], dtype=tf.string,
                    output_shape=[512], trainable=False)]
    )
    # Convert all texts to vectors
    text_features = text_model.predict(texts, batch_size=BATCH_SIZE, use_multiprocessing=True, workers=-1)
    print('\nText features extracted. Shape:', text_features.shape)
    # Get a list of similar title pairs
    title_pairs = feature_mining(text_features, query_chunk_size=1000, top_k=100)
    # Convert list to pd.DataFrame
    title_pairs = pd.DataFrame(title_pairs, columns=['text_score', 'idx_1', 'idx_2'])
    text_features = text_features.astype(np.float16, copy=False)
    # Order index pairs so that first index is lower than the second one
    # to avoid duplicates when merging text pairs with image pairs
    ordered = title_pairs[['idx_1', 'idx_2']].agg(['min', 'max'], axis='columns')
    title_pairs[['idx_1', 'idx_2']] = ordered.values
    print(f'Number of pairs: {len(title_pairs)}')
    # Minimum similarity score among title pairs returned by a text model
    text_sim_threshold = title_pairs['text_score'].min()
    print('Minimum text similarity score:', text_sim_threshold)
    return text_features, title_pairs, text_sim_threshold


def pytorch_cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a, b)
    :param a: input data as a tensor or np.array
    :param b: input data as a tensor or np.array
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def feature_mining(features: np.array,
                   query_chunk_size: int = 5000,
                   corpus_chunk_size: int = 100000,
                   max_pairs: int = 500000,
                   top_k: int = 100) -> list:
    """Given an array of image or text features, this function performs data mining.
    It compares all text / images against all other texts / images and returns a list
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

            cos_scores = pytorch_cos_sim(features[query_start_idx:query_end_idx],
                                         features[corpus_start_idx:corpus_end_idx]).cpu()

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


def get_similar_items(img_paths: pd.Series,
                      texts: pd.Series,
                      reference_df: pd.DataFrame,
                      labels=False) -> tuple:
    """Function produces a combined DataFrame of most similar item pairs
    with high text and / or image similarity scores.
    :param img_paths: pd.Series containing paths to files with images
    :param texts: pd.Series containing item titles
    :param reference_df: original DataFrame with items
    :param labels: Boolean argument to indicate if we know actual labels for this data set
    :returns: Tuple where 0-th element contains DataFrame with columns=['idx_1', 'idx_2',
    'image_score', 'text_score', 'txt_img_score'],
     1-st element is minimum image score before merging, 2-nd element is minimum text score before merging
    """
    # Process image paths
    image_features, pairs, image_sim_threshold = get_image_features(paths=img_paths)
    # Process titles
    text_features, title_pairs, text_sim_threshold = get_text_features(texts=texts)

    # Combine similarity scores for texts and images
    pairs = join_pairs(pairs, title_pairs, reference_df=reference_df, check_labels=labels)

    # Use Dot layer to calculate cosine similarity
    cosine_model = tf.keras.layers.Dot(axes=1, normalize=True, dtype='float16')

    # Fill in missing scores for text and image similarity in chunks of size 100,000
    chunk = 100_000

    # Replace NaNs in image scores
    while True:
        if pairs['image_score'].isna().sum() == 0:
            break
        else:
            replace_idx = pairs[pairs['image_score'].isna()].head(chunk).index
            # Indexes for image pairs with missing image similarity scores
            idx_1 = pairs[pairs['image_score'].isna()].head(chunk)['idx_1'].values
            idx_2 = pairs[pairs['image_score'].isna()].head(chunk)['idx_2'].values
            # Two sequences of image features for pairs with missing image similarity score
            features_1 = image_features[idx_1]
            features_2 = image_features[idx_2]
            # Pass both arrays through Dot layer to get cosine similarity scores
            pairs.loc[replace_idx, 'image_score'] = cosine_model([features_1, features_2]).numpy()

    # Replace NaNs in text scores
    while True:
        if pairs['text_score'].isna().sum() == 0:
            break
        else:
            replace_idx = pairs[pairs['text_score'].isna()].head(chunk).index
            # Indexes for text pairs with missing text similarity score
            idx_1 = pairs[pairs['text_score'].isna()].head(chunk)['idx_1'].values
            idx_2 = pairs[pairs['text_score'].isna()].head(chunk)['idx_2'].values
            # Two sequences of text features for pairs with missing similarity scores
            features_1 = text_features[idx_1]
            features_2 = text_features[idx_2]
            # Pass both arrays through Dot layer to get cosine similarity scores
            pairs.loc[replace_idx, 'text_score'] = cosine_model([features_1, features_2]).numpy()

    # Add a combined text and image similarity scores
    pairs['txt_img_score'] = pairs['text_score'] * pairs['image_score']

    return pairs, image_sim_threshold, text_sim_threshold


def join_pairs(images: pd.DataFrame, texts: pd.DataFrame,
               reference_df: pd.DataFrame,
               check_labels=False) -> pd.DataFrame:
    """Function joins text pairs and image pairs into combined DataFrame,
    adds known phash pairs (for train and test set) and known label group pairs (for train set).
    :param images: image pairs with high similarity scores
    :param texts: title pairs with high similarity scores
    :param reference_df: original DataFrame with items
    :param check_labels: boolean flag, if True - add indexes for all known title pairs
           from the train set regardless of similarity scores and phash values
    :return: Returns unified DataFrame with columns ['idx_1', 'idx_2', 'text_score', 'image_score']
    """
    # Merge text features with image features on index pairs
    combined = pd.merge(texts, images, how='outer')
    print(f'\nMerging {len(texts)} text pairs with {len(images)} image pairs.\n'
          f'Result: {len(combined)} pairs')

    # Add all remaining pairs with identical phash
    pairs = []
    groups = reference_df.groupby(by='image_phash')
    # Numeric index of rows in train set as a temporary column
    reference_df['num_idx'] = [idx for idx in range(len(reference_df))]
    for group in groups.indices:
        num_index = groups.get_group(group)['num_idx']
        # All combinations of index pairs with the same phash
        pairs = list(itertools.combinations(num_index, 2))
        pairs.extend(pairs)
    pairs = pd.DataFrame(pairs, columns=['idx_1', 'idx_2'])
    combined = pd.merge(combined, pairs, how='outer')
    print(f'\nAdded phash pairs. Result: {len(combined)}')

    # When dealing with train set, add all remaining item pairs
    # with the same label group regardless of similarity scores.
    if check_labels:
        pairs = []
        groups = reference_df.groupby(by='label_group')
        for group in groups.indices:
            num_index = groups.get_group(group)['num_idx']
            # All combinations of index pairs with the same label group
            pairs = list(itertools.combinations(num_index, 2))
            pairs.extend(pairs)
        pairs = pd.DataFrame(pairs, columns=['idx_1', 'idx_2'])
        combined = pd.merge(combined, pairs, how='outer')
        print(f'Added all known label group pairs. Result: {len(combined)}')

    return combined


def get_tfidf_similarity(pairs_df: pd.DataFrame, texts: pd.Series) -> np.array:
    """Function calculates similarity scores for title pairs
    based on the TF-IDF vectors.
    :param pairs_df: DataFrame with indexes of candidate item pairs
    :param texts: Series of titles from the original DataFrame
    :return: Array with TF-IDF similarity scores for candidate item pairs
    """

    # Transform all titles into TF-IDF matrix
    vectorizer = TfidfVectorizer(decode_error='ignore',
                                 stop_words='english',
                                 max_features=5_000)
    vectors = vectorizer.fit_transform(texts).toarray().astype(np.float16, copy=False)
    print('TF-IDF features shape:', vectors.shape)

    # Use Dot layer to calculate cosine similarity
    cosine_model = tf.keras.layers.Dot(axes=1, normalize=True, dtype='float16')

    # Calculate similarity scores for title pairs in chunks of size 15,000
    chunk = 15_000
    start_idx = 0
    last_idx = len(pairs_df) - 1

    pairs_df['tf_idf_score'] = np.nan

    while start_idx <= last_idx:
        # End row index of current chunk
        end_idx = start_idx + chunk - 1
        if end_idx > last_idx:
            end_idx = last_idx

        # Indexes in the matrix for 1st and 2nd title in each pair
        idx_1 = pairs_df.loc[start_idx:end_idx, 'idx_1'].values
        idx_2 = pairs_df.loc[start_idx:end_idx, 'idx_2'].values

        # Two sequences of TF-IDF features for pairs in current chunk
        features_1 = vectors[idx_1]
        features_2 = vectors[idx_2]
        # Pass both arrays through Dot layer to get cosine similarity scores
        pairs_df.loc[start_idx:end_idx, 'tf_idf_score'] = cosine_model([features_1, features_2]).numpy()
        # Update start row index for the next iteration
        start_idx += chunk

    # In case any of the TF-IDF values were all zeros
    print(f'Calculated TF-IDF scores for {last_idx + 1} title pairs.')
    print(f'{pairs_df["tf_idf_score"].isna().sum()} TF-IDF scores were undefined.')
    pairs_df['tf_idf_score'] = pairs_df['tf_idf_score'].fillna(0)

    return pairs_df['tf_idf_score'].values


def check_identity(pairs_df: pd.DataFrame, reference_df: pd.DataFrame, par: str) -> pd.Series:
    """Function finds values in 'par' column of 'reference_df'
    using 2 row indexes from 'df' and returns a binary column
    for match between two items (1 - matching, 0 - non-matching).
    :param pairs_df: DataFrame with indexes of candidate pairs
    :param reference_df: DataFrame with original data
    :param par: parameter to check
    :return Series with binary values - result of the comparison
    """
    # Temporary dataframe to compare 'par' values for item pairs
    identity = pd.DataFrame()
    # Look up values by respective row indexes
    identity['item_1'] = reference_df.iloc[pairs_df['idx_1'], :][par].values
    identity['item_2'] = reference_df.iloc[pairs_df['idx_2'], :][par].values
    # Binary column signifying match or mismatch
    identity['match'] = (identity['item_1'] == identity['item_2']).astype('int')
    return identity['match']


def numbers_identity(pairs_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.Series:
    """Function extracts numbers from titles and compares them,
    returns a binary column for match between numbers in two titles.
    :param pairs_df: DataFrame with candidate pairs
    :param reference_df: DataFrame with original data
    :return Series with binary values - result of the comparison
    """
    # Temporary dataframe to compare numbers for item pairs
    identity = pd.DataFrame()
    # Look up values by respective row indexes
    identity['item_1'] = reference_df.iloc[pairs_df['idx_1'], :]['title'].values
    identity['item_2'] = reference_df.iloc[pairs_df['idx_2'], :]['title'].values
    # Extract numbers and convert them to space-delimited string
    identity['nums_1'] = identity['item_1'].apply(lambda x: ' '.join(re.findall(r'\d+', x)))
    identity['nums_2'] = identity['item_2'].apply(lambda x: ' '.join(re.findall(r'\d+', x)))
    # Binary column signifying match or mismatch
    identity['match'] = (identity['nums_1'] == identity['nums_2']).astype('int')
    return identity['match']


def add_binary_features(pairs: pd.DataFrame, reference_df: pd.DataFrame, labels=False) -> pd.DataFrame:
    """Function adds binary feature columns with 0/1 values to candidates DataFrame
    based on label groups (if known), phash match and match of numbers in titles,
    creates additional features by multiplying similarity score and binary features
    with significance coefficients.
    :param pairs: Candidate pairs of similar items
    :param reference_df: Original pd.DataFrame with items
    :return: Candidate pairs with new feature columns added
    """
    if labels:
        # Pair belongs to the same label group
        pairs['true_match'] = check_identity(pairs[['idx_1', 'idx_2']], reference_df, 'label_group')
    # Same phash
    pairs['phash_match'] = check_identity(pairs[['idx_1', 'idx_2']], reference_df, 'image_phash')
    # Identical numbers in titles
    pairs['nums_match'] = numbers_identity(pairs[['idx_1', 'idx_2']], reference_df)
    return pairs


def abs_path(file_name: str, directory: str) -> str:
    """Function returns a Series of absolute paths to images
    given file names and directory name.
    :param file_name: Name of the image file
    :param directory: Name of directory containing the file
    :return Path to the image file
    """
    return os.path.join(directory, file_name)


def process_path(file_path: str):
    """Function converts a path to file into preprocessed image.
    :param file_name: Name of the image file
    :return Tensor with preprocessed image from the file
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.keras.applications.efficientnet.preprocess_input(img)


def configure_for_performance(ds):
    """Function applies batches and prefetches dataset
    to optimize data processing.
    :param ds: TensorFlow Dataset object
    :return Batched TensorFlow Dataset object with prefetch() applied
    """
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def analyze_similarities(pairs: pd.DataFrame,
                         text_sim_threshold: float,
                         image_sim_threshold: float):
    """Function compares number of actual item pairs in pairs
    produced by language and image models.
    :param pairs: DataFrame with similar item pairs
    :param text_sim_threshold: Minimum cosine similarity among pairs produced by language model
    :param image_sim_threshold: Minimum cosine similarity among pairs produced by image model
    """
    # Check the results of text model
    text_pairs = pairs[pairs['text_score'] >= text_sim_threshold]
    print(f'Number of similar title pairs returned by language model: {len(text_pairs)}')
    print('Distribution of True and False predictions:')
    print(text_pairs['true_match'].value_counts(normalize=True))

    # Check the results of image model
    image_pairs = pairs[pairs['image_score'] >= image_sim_threshold]
    print(f'\nNumber of similar image pairs returned by image model: {len(image_pairs)}')
    print('Distribution of True and False predictions:')
    print(image_pairs['true_match'].value_counts(normalize=True))

    # Check how many actual title combinations are there in the train set
    n_similar = data['label_group'].value_counts()
    n_similar = pd.DataFrame(n_similar.values, index=n_similar.index, columns=['n_samples'])
    n_similar['n_combinations'] = n_similar['n_samples'].apply(combinations)
    print(f'\nNumber of actual item pairs in the train set: {n_similar["n_combinations"].sum()}\n')

    # Check distribution of True and False predictions for various text similarity scores
    for thr in (0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        print('-' * 50)
        print(f'Text cosine similarity score over {thr}')
        pairs_sample = pairs[pairs['text_score'] >= thr]
        print(f'Number of similar title pairs: {len(pairs_sample)}')
        print(pairs_sample['true_match'].value_counts(normalize=True))

    # Check distribution of True and False predictions for various TF-IDF similarity scores
    for thr in (0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        print('-' * 50)
        print(f'TF-IDF cosine similarity score over {thr}')
        pairs_sample = pairs[pairs['tf_idf_score'] >= thr]
        print(f'Number of similar title pairs: {len(pairs_sample)}')
        print(pairs_sample['true_match'].value_counts(normalize=True))

    # Check distribution of True and False predictions for various image similarity scores
    for thr in (0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99):
        print('-' * 50)
        print(f'Image cosine similarity score over {thr}')
        pairs_sample = pairs[pairs['image_score'] >= thr]
        print(f'Number of similar image pairs: {len(pairs_sample)}')
        print(pairs_sample['true_match'].value_counts(normalize=True))

    # Check distribution of True and False predictions for various levels in txt_img_score
    for thr in (0.6, 0.7, 0.8, 0.9, 0.95):
        print('-' * 50)
        print(f'txt_img_score over {thr}')
        pairs_sample = pairs[pairs['txt_img_score'] >= thr]
        print(f'Number of similar item pairs: {len(pairs_sample)}')
        print(pairs_sample['true_match'].value_counts(normalize=True))


def analize_binaries(pairs: pd.DataFrame):
    """Function prints distribution of True and False predictions
    for binary features (identity of phash and numbers in title).
    :param pairs: DataFrame of candidate item pairs
    """
    # Check if identical phash can be used to improve the accuracy
    same_phash = pairs[pairs['phash_match'] == 1]
    different_phash = pairs[pairs['phash_match'] == 0]

    print('For item pairs with the same phash:')
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


def add_color_features(pairs_df: pd.DataFrame,
                       reference_df: pd.DataFrame) -> pd.DataFrame:
    """Function adds average color features to DataFrame with candidate pairs
    and creates additional features by multiplying similarity scores.
    :param pairs_df: Candidate item pairs_df
    :param reference_df: DataFrame with original data
    :return: DataFrame with candidate item pairs_df with color features added
    """
    # Image paths in the same order as listed in the train set
    input_data = tf.data.Dataset.from_tensor_slices(reference_df['image'])
    # Preprocess images
    input_data = input_data.map(avg_image_color, num_parallel_calls=AUTOTUNE)
    input_data = configure_for_performance(input_data)
    # Create a nominal model:
    color_model = tf.keras.models.Sequential([tf.keras.layers.Layer(input_shape=(3,), dtype=tf.float16)])
    # Features for all images: n_samples x 3 columns (RGB channels)
    image_colors = pd.DataFrame(color_model.predict(input_data))
    # Calculate absolute differences for image pairs_df
    difference = color_diffs(pairs_df[['idx_1', 'idx_2']], image_colors)
    pairs_df = pd.concat([pairs_df, difference], axis='columns')
    # We use average color difference across all channels to get a new feature
    # combining text and image similarity scores and color difference:
    pairs_df['color_diff'] = (pairs_df[['R_diff', 'G_diff', 'B_diff']]).mean(axis='columns')
    pairs_df['combined_score'] = pairs_df['txt_img_score'] * (1 - pairs_df['color_diff'])

    # Create additional features
    pairs_df['combined_phash'] = pairs_df['combined_score']
    pairs_df.loc[pairs_df['phash_match'] == 0, 'combined_phash'] = 0.6 * pairs_df.loc[
        pairs_df['phash_match'] == 0, 'combined_score']
    pairs_df['combined_phash_nums'] = pairs_df['combined_phash']
    pairs_df.loc[pairs_df['nums_match'] == 0, 'combined_phash_nums'] = 0.9 * pairs_df.loc[
        pairs_df['nums_match'] == 0, 'combined_phash']
    pairs_df['txt_img_idf_score'] = pairs_df['txt_img_score'] * pairs_df['tf_idf_score']
    pairs_df['combined_idf_score'] = pairs_df['combined_score'] * pairs_df['tf_idf_score']

    return pairs_df


@tf.function
def avg_image_color(path: str):
    """Function reads image from file and returns
    average color values for 3 channels.
    :param path: File path to an image
    :return Average color values (tensor with 3 values)
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.reduce_mean(img, axis=(0, 1))


def color_diffs(pairs_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Function calculates percentage differences
    between average color values for image pairs.
    :param pairs_df: DataFrame with indexes of candidate pairs
    :param reference_df: DataFrame with original data
    :return DataFrame with 3 columns ['R_diff', 'G_diff', 'B_diff']
    """
    # Temporary DataFrames for comparison
    img_1 = reference_df.iloc[pairs_df['idx_1'], :].reset_index(drop=True)
    img_2 = reference_df.iloc[pairs_df['idx_2'], :].reset_index(drop=True)
    img_1.columns = ['R1', 'G1', 'B1']
    img_2.columns = ['R2', 'G2', 'B2']
    averages = pd.concat([img_1, img_2], axis='columns')
    # Calculate differences for 3 color channels
    for color in ('R', 'G', 'B'):
        averages[f'{color}_diff'] = (averages[f'{color}1'] - averages[f'{color}2']) / 255
        averages[f'{color}_diff'] = averages[f'{color}_diff'].abs()
        averages.drop([f'{color}1', f'{color}2'], axis='columns', inplace=True)
    return averages


def show_image_pair(path_1: str, path_2: str):
    """Function shows a pair of images.
    :param path_1: File path to an image
    :param path_2: File path to a second image
    """
    img_1 = tf.io.read_file(path_1)
    img_1 = tf.image.decode_jpeg(img_1, channels=3)
    img_1 = tf.image.resize(img_1, [IMG_SIZE, IMG_SIZE])
    img_2 = tf.io.read_file(path_2)
    img_2 = tf.image.decode_jpeg(img_2, channels=3)
    img_2 = tf.image.resize(img_2, [IMG_SIZE, IMG_SIZE])

    fig = plt.gcf()
    fig.set_size_inches(8, 16)
    plt.subplot(1, 2, 1)
    plt.imshow(img_1)
    plt.axis('Off')
    plt.subplot(1, 2, 2)
    plt.imshow(img_2)
    plt.axis('Off')
    plt.show()


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


def get_confusion_matrix(clf, y_test, y_pred):
    """Function plots a confusion matrix for given prediction
    and ground truth values.
    :param clf: Classification model
    :param y_test: Ground-truth values
    :param y_pred: Predicted values
    """
    conf_matrix = confusion_matrix(y_test, y_pred)

    target_names = clf.classes_
    columns = [f'pred_{name}' for name in target_names]
    indexes = [f'actual_{name}' for name in target_names]
    conf_matrix = pd.DataFrame(conf_matrix, columns=columns, index=indexes)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdBu_r')
    plt.yticks(rotation=0)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def score_model(clf, X_train, y_train, X_test, y_test, skf, probs=True):
    """Function evaluates classifier performance using cross-validation
    and calculating metrics for test data (ROC score, confusion matrix,
    classification report).
    :param clf: Classification model
    :param X_train: Train input data
    :param y_train: Train labels
    :param X_test: Test input data
    :param y_test: Test labels
    :param skf: StratifiedKFold object to use in cross-validation
    :param probs: Boolean argument defining whether or not this classifier outputs probabilities
    """
    # Performance on train data
    scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='f1')
    print('Cross-validation F1 score:', scores.mean())

    # Performance on data, which was left out during training
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # If classifier provides probability estimates
    if probs:
        y_pred_prob = clf.predict_proba(X_test)
        ROC_score = roc_auc_score(y_test, y_pred_prob[:, 1], multi_class='ovo')
        print('ROC score on test data:', ROC_score)

    get_confusion_matrix(clf, y_test, y_pred)
    print(classification_report(y_test, y_pred))


def find_image_pair(idx_1: int, idx_2: int, reference_df: pd.DataFrame, directory: str):
    """Function finds paths to image pair by row indexes and shows images.
    :param idx_1: Numeric row index of an item
    :param idx_2: Numeric row index of the second item
    :param reference_df: DataFrame with original data
    :param directory: Directory with image files
    """
    file_1, file_2 = reference_df.loc[[idx_1, idx_2], 'image']
    path_1 = abs_path(file_1, directory)
    path_2 = abs_path(file_2, directory)
    show_image_pair(path_1, path_2)


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


# --------------------------- Processing train set --------------------------------

# Load train data
data = get_csv_data(csv_path=train_csv_path, img_dir=train_img_path)

# Extracting features from images and titles and searching for similar item pairs
pairs, image_sim_threshold, text_sim_threshold = get_similar_items(img_paths=data['image'],
                                                                   texts=data['title'],
                                                                   reference_df=data,
                                                                   labels=True)

# Launch garbage collector to avoid out-of-memory error
gc.collect()

# Add similarity scores based on TF-IDF
pairs['tf_idf_score'] = get_tfidf_similarity(pairs[['idx_1', 'idx_2']].copy(),
                                             data['title'])
gc.collect()

# See some examples of highly similar image pairs
examples = pairs[pairs['image_score'] > 0.99].head()[['idx_1', 'idx_2']].values
for idx_1, idx_2 in examples:
    find_image_pair(idx_1, idx_2, data, train_img_path)
    print(data.loc[[idx_1, idx_2], ['title', 'label_group']])

# See some examples of less similar image pairs
examples = pairs[pairs['image_score'] < image_sim_threshold + 0.02].head()[['idx_1', 'idx_2']].values
for idx_1, idx_2 in examples:
    find_image_pair(idx_1, idx_2, data, train_img_path)
    print(data.loc[[idx_1, idx_2], ['title', 'label_group']])

# See some examples of highly similar title pairs
for i in range(5):
    examples = pairs[pairs['text_score'] > 0.98]
    score = examples.iloc[i, :]['text_score']
    idx_1, idx_2 = examples.iloc[i, :][['idx_1', 'idx_2']]
    title_1 = data.loc[idx_1, 'title']
    title_2 = data.loc[idx_2, 'title']
    print('-' * 50)
    print(title_1)
    print(title_2)
    print(f'Similarity score = {score}')

# See some examples of less similar title pairs
for i in range(5):
    examples = pairs[pairs['text_score'] < text_sim_threshold + 0.02]
    score = examples.iloc[i, :]['text_score']
    idx_1, idx_2 = examples.iloc[i, :][['idx_1', 'idx_2']]
    title_1 = data.loc[idx_1, 'title']
    title_2 = data.loc[idx_2, 'title']
    print('-' * 50)
    print(title_1)
    print(title_2)
    print(f'Similarity score = {score}')

# Add binary features (phash match and numbers match)
# and ground truth labels for training
pairs = add_binary_features(pairs, data, labels=True)

# See how many actual label pairs were returned by language and image models
analyze_similarities(pairs, text_sim_threshold, image_sim_threshold)

# Check mislabeled pairs: high text similarity scores and different label groups
examples = pairs[(pairs['text_score'] >= 0.95) & (pairs[pairs['true_match'] == 0])].head()[['idx_1', 'idx_2']].values
for idx_1, idx_2 in examples:
    find_image_pair(idx_1, idx_2, data, train_img_path)
    print(data.loc[[idx_1, idx_2], ['title', 'label_group']])

# Check mislabeled pairs: high image similarity score and different label groups
examples = pairs[(pairs['image_score'] >= 0.98) & (pairs[pairs['true_match'] == 0])].head()[['idx_1', 'idx_2']].values
for idx_1, idx_2 in examples:
    find_image_pair(idx_1, idx_2, data, train_img_path)
    print(data.loc[[idx_1, idx_2], ['title', 'label_group']])

# See how binary features affect predictions
analize_binaries(pairs)

# Add color features (average colors) and additional features
# derived from multiplying similarity scores
pairs = add_color_features(pairs, data)
print(pairs.head())

# Target values and features that could be considered to thain the classifier
target = ['true_match']
features = ['text_score', 'tf_idf_score', 'phash_match', 'nums_match',
            'image_score', 'txt_img_score', 'R_diff', 'G_diff', 'B_diff', 'color_diff',
            'combined_score', 'combined_phash', 'combined_phash_nums',
            'txt_img_idf_score', 'combined_idf_score']

# Visualize correlation between features and target
# and distribution of values in two classes
correlation_and_distribution(pairs, features, target)

# Update the list of features removing excessive ones
features = ['text_score', 'tf_idf_score', 'phash_match', 'nums_match',
            'image_score', 'txt_img_score', 'color_diff', 'combined_score',
            'combined_phash', 'combined_phash_nums', 'txt_img_idf_score', 'combined_idf_score']

# Use 3 classifiers with different architectures and different recall/precision ratios
clf_NB = GaussianNB(var_smoothing=1e-07)
clf_RF = RandomForestClassifier(min_samples_leaf=10, n_jobs=-1)
clf_KNN = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, n_jobs=-1))

clf = VotingClassifier(estimators=[('NB', clf_NB), ('RF', clf_RF), ('KNN', clf_KNN)],
                       voting='hard', n_jobs=-1)

# If we run this notebook for cross-validation (when submitting, it's not used)
if scoring:
    skf = StratifiedKFold(5, shuffle=True, random_state=1)
    scores = cross_val_score(clf, pairs[features], pairs[target].values.ravel(), cv=skf, scoring='f1')
    print('Cross-validation scores:', scores)
    print('Average train CV score:', np.mean(scores))
    # Save train pairs to use for quicker experiments
    pairs.to_csv('train_pairs.csv', index=False)

# Train the ensemble using all available data
clf.fit(
    pairs[features],
    pairs[target].values.ravel()
)

# ----------------------------- Applying models to test set ---------------------------

# Load test data
data = get_csv_data(csv_path=test_csv_path, img_dir=test_img_path)

# Extracting features from images and titles and searching for similar item pairs
pairs, _, _ = get_similar_items(img_paths=data['image'], texts=data['title'], reference_df=data)

# Launch garbage collector to avoid out-of-memory error
gc.collect()

# Add similarity scores based on TF-IDF
pairs['tf_idf_score'] = get_tfidf_similarity(pairs[['idx_1', 'idx_2']].copy(),
                                             data['title'])
gc.collect()

# Add binary features and multiplicative features
pairs = add_binary_features(pairs, data)

# Add color features
pairs = add_color_features(pairs, data)
print(pairs.head())

# Predict matching pairs with classification model
pairs['pred_match'] = clf.predict(pairs[features])

# Drop all pairs that did not pass the final classifier
pairs = pairs[pairs['pred_match'] == 1]

# Add a column with space-delimited IDs for matching items to the test data set
data['matches'] = find_similar_items(data, pairs)

# Save two required columns to csv file
data[['posting_id', 'matches']].to_csv('/kaggle/working/submission.csv', index=False)
