"""Utility functions for using in book recommendation algorithms.
Data is loaded from 3 csv files (for books, users and ratings).
Original data includes user ID, age and location, book author,
title, publisher, year of publication and ISBN,
book ratings by users (1 to 10 or 0 if user did not rate the book),
URLs to cover images in 3 sizes.
Data preprocessing steps:
- Correction of technical errors and misspellings in the original data
  (incorrect year of publication, misplaced column values, outlier values
  in user age, nonconventional descriptions of user location).
- Reindexing of books. Various editions of the same book have different ISBNs.
  Unique content IDs help to avoid recommending the same content
  (same books published in different periods and by different publishers)
  to users who already rated or read it.
- Merging all data into a single DataFrame.
"""

import pandas as pd
import numpy as np
import string
import random
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# Display settings
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 14})

# Variable to store title embeddings for repeated queries
title_embeddings = None

# File paths
books_path = './data/Books.csv'
ratings_path = './data/Ratings.csv'
users_path = './data/Users.csv'

# ------------------------------------ Functions ------------------------------------


def get_data() -> pd.DataFrame:
    """Function extracts data for books, ratings and users
    from csv files, cleans data and joins data on ISBN and user ID.
    Creates a global variable "books", that contains lookup dictionary
    to find author and book title by content ID.
    :return: Combined pd.DataFrame with all available data
    """
    global books  # Global dict to look up authors and titles by content IDs

    # Original data
    books_df = pd.read_csv(books_path, low_memory=False)
    books_df = clean_book_data(books_df)
    print(f'Books data extracted: {len(books_df)} rows')

    # Various editions of the same book have different ISBNs.
    # We reindex the books so that identical title-author pairs
    # get unique key, which will be used to look up content for users
    # and estimate ratings and number of readings.
    books_df = reindex_books(books_df)

    ratings_df = pd.read_csv(ratings_path, low_memory=False)
    print(f'Ratings data extracted: {len(ratings_df)} rows')

    users_df = pd.read_csv(users_path, low_memory=False)
    users_df = clean_users_data(users_df)
    print(f'Users data extracted: {len(users_df)} rows')

    df = pd.merge(books_df, ratings_df, on='ISBN')
    print(f'Books and ratings data combined: {len(df)} rows')
    df = pd.merge(df, users_df, on='User-ID')
    print(f'Books, ratings and users data combined: {len(df)} rows')

    # Assign values to the lookup dictionary
    # (keys = content IDs, values = {"Book-Title": ..., "Book-Author": ...})
    books_df = books_df.drop_duplicates(subset=['Content_ID']).set_index('Content_ID', drop=True)
    books = books_df[['Book-Title', 'Book-Author']].to_dict('index')

    return df


def clean_book_data(df: pd.DataFrame) -> pd.DataFrame:
    """Function corrects errors and dtypes in books DataFrame
    and fills in missing values.
    :param df: Original DataFrame extracted from csv file
    :return: Updated DataFrame
    """
    # In several rows title column contains both title and author
    # while next columns are shifted one step to the left.
    errors_idx = df[df['Year-Of-Publication'].apply(len) > 4].index  # Filter by long string in year column
    values_to_move = df.loc[errors_idx, 'Book-Author':'Image-URL-M'].values
    df.loc[errors_idx, 'Year-Of-Publication':'Image-URL-L'] = values_to_move  # Move values one step to the right
    splitted_text = df.loc[errors_idx, 'Book-Title'].str.split(';').apply(pd.Series)
    df.loc[errors_idx, 'Book-Title':'Book-Author'] = splitted_text.values  # Split title and author
    # Convert year column from object to integer
    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')
    # For some books year of publication is incorrect (2030, 2050, etc.).
    # Assume that it should be 1930, 1950, etc. Zero values are replaced by 2000.
    df['Year-Of-Publication'] = df['Year-Of-Publication'].apply(
        lambda x: x - 100 if x > 2020 else (2000 if x == 0 else x)
    )
    # Several missing values for author and publisher
    df.fillna('N/A', inplace=True)
    # Transform each author name to capitalized format
    df['Book-Author'] = df['Book-Author'].apply(lambda x: x.lower().title())
    # Replace ampersands and capitalize book titles
    df['Book-Title'] = df['Book-Title'].str.replace('&amp;', 'and').apply(lambda x: x.lower().title())

    return df


def clean_users_data(df: pd.DataFrame) -> pd.DataFrame:
    """Function adds country column to users data
    and groups users by age.
    :param df: Original DataFrame extracted from csv file
    :return: Updated DataFrame
    """
    # Extract the last value, which in most cases is the country name.
    df['Country'] = df['Location'].apply(lambda x: x.split(',')[-1].strip())
    # Country name contains misspellings and errors.
    # We assume that names repeating 10 time and more
    # are correct country identifiers.
    countries = df['Country'].value_counts()
    countries = set(countries[countries >= 10].index)
    df['Country'] = df['Country'].apply(lambda x: x if x in countries else 'other')
    df['Country'] = df['Country'].fillna('other')
    # Remove punctuation
    df['Country'] = df['Country'].apply(
        lambda x: x.translate(str.maketrans('', '', string.punctuation))
    )
    df['Country'] = df['Country'].replace(' na', 'other').replace('na', 'other').replace('', 'other')

    # Age column contains large number of outliers and missing values.
    # Group users by age assuming that extreme values on both sides
    # will merge into respective groups and will not affect the accuracy.
    # Introduce a separate group for users of unknown age.
    df['Age'] = df['Age'].fillna(-1)
    df['Age_group'] = pd.cut(df['Age'], bins=[-2, 0, 6, 10, 15, 20, 35, 55, 250],
                             labels=['unknown', 'preschool', 'primary_school',
                                     'secondary_school', 'teenager', 'young_adult',
                                     'adult', 'senior'])

    return df


def reindex_books(df: pd.DataFrame) -> pd.DataFrame:
    """Function creates 'Content_ID' column with a unique numeric key
    for all identical author-title pairs in the books DataFrame.
    'Content_ID' helps to identify identical content (same books
    published in different years or by different publishers and having different ISBNs).
    :param df: DataFrame with all available data
    :return: Updated DataFrame
    """
    # Drop author-title duplicates from the original DataFrame
    # and create a dictionary where each author-title string
    # corresponds to a unique numeric value.
    unique = df[['Book-Title', 'Book-Author']].drop_duplicates().reset_index(drop=True).copy()
    unique['pair'] = unique['Book-Author'] + unique['Book-Title']
    print(f'Books reindexed: {len(unique)} unique content IDs created')
    unique = dict(zip(unique['pair'], unique.index))
    # Add a new column with unique key to the original DataFrame
    df['Content_ID'] = (df['Book-Author'] + df['Book-Title']).apply(lambda x: unique[x])
    return df


def recommend_by_ratings(df: pd.DataFrame,
                         countries=None,
                         age_groups=None,
                         n_books: int = 10,
                         threshold: int = 10) -> list:
    """Function estimates average ratings per book
    and produces recommendations based on book ratings
    taking into account the threshold - minimum number of readers
    necessary to trust the average rating.
    :param df: DataFrame with all available data
    :param countries: Optional argument to select geographic location
    (a list of countries should be passed)
    :param age_groups: Optional argument to select age groups
    (a list of groups should be passed)
    :param n_books: Maximum number of books to recommend
    :param threshold: Minimum required number of ratings per book to consider
    :return: List of recommended content IDs (if both threshold and n_books are high,
    total number of recommended books could be smaller than n_books)
    """
    # If list of countries is passed, limit the database to selected region
    if countries:
        df = df[df['Country'].isin(countries)]

    # If list of age groups is passed, limit the database to selected demographics
    if age_groups:
        df = df[df['Age_group'].isin(age_groups)]

    # Drop all rows where rating=0 (user did not rate the book)
    df = df[df['Book-Rating'] > 0]

    # Calculate average rating and number of ratings per book
    avg_ratings = df.groupby(by='Content_ID')['Book-Rating'].agg(['mean', 'count'])

    # Drop books with small number of ratings
    avg_ratings = avg_ratings[avg_ratings['count'] >= threshold]

    # Get content IDs for books with the highest average ratings
    avg_ratings.sort_values(by='mean', ascending=False, inplace=True)
    recommendations = avg_ratings.head(n_books).index.to_list()
    display_recommendations(recommendations)

    return recommendations


def recommend_by_readings(df: pd.DataFrame,
                          countries=None,
                          age_groups=None,
                          n_books: int = 10) -> list:
    """Function estimates total number of reading per book
    and returns a list of content IDs for books with largest readership base.
    :param df: DataFrame with all available data
    :param countries: Optional argument to select geographic location
    (a list of countries should be passed)
    :param age_groups: Optional argument to select age groups
    (a list of groups should be passed)
    :param n_books: Maximum number of books to recommend
    :return: List of recommended content IDs
    """
    # If list of countries is passed, limit the database to selected region
    if countries:
        df = df[df['Country'].isin(countries)]

    # If list of age groups is passed, limit the database to selected demographics
    if age_groups:
        df = df[df['Age_group'].isin(age_groups)]

    # Count total number of readings per book
    # regardless of whether the user gave any rating or not.
    readings = df.groupby(by='Content_ID')['Book-Rating'].count()
    recommendations = readings.sort_values(ascending=False).head(n_books).index.to_list()
    display_recommendations(recommendations)

    return recommendations


def display_recommendations(content: list):
    """Function prints a list of recommended books.
    :param content: List of content IDs
    """
    print('Recommended books:')
    for i, ID in enumerate(content):
        print(f'{i + 1}. {books[ID]["Book-Author"]}: {books[ID]["Book-Title"]}')


def select_user(df: pd.DataFrame):
    """Function selects a random user.
    :param df: DataFrame with all available data
    :return: User ID
    """
    user = random.choice(df['User-ID'])
    print(f'Selected user: ID = {user}')
    return user


def get_user_data(df: pd.DataFrame, user, titles_only=False) -> tuple:
    """Function extracts user's data by user ID, filters out disliked books
    and returns either a list of book titles or a tuple with authors names
    and content IDs.
    :param df: DataFrame with all available data
    :param user: User ID
    :param titles_only: Boolean parameter specifying if the function
    should return only a list of book titles
    :return: Tuple with 2 elements (array of all authors read by user,
    set of all books IDs) or a list of all book titles sorted by ratings
    """
    user_data = df[df['User-ID'] == user].copy()
    print(f'User activity: {len(user_data)} logs')

    # Drop obviously disliked items rated between 1 and 4
    user_data = user_data[(user_data['Book-Rating'] == 0)
                          | (user_data['Book-Rating'] > 4)]
    print(f'Without disliked books user history contains {len(user_data)} items')

    # Return only book titles
    if titles_only:
        user_data.sort_values(by='Book-Rating', ascending=False, inplace=True)
        return user_data['Book-Title'].to_list()

    # Books IDs and authors read by user
    book_ids = set(user_data['Content_ID'])
    authors = user_data['Book-Author'].unique()

    return authors, book_ids


def recommend_by_activity(df: pd.DataFrame, user_id=None, limit=10):
    """Function selects books by authors most read by the user,
    with the exception of books user already read.
    :param df: DataFrame with all available data
    :param user_id: Optional argument for user ID (if not provided, will be randomly selected)
    :param limit: Optional argument to limit total number of recommendations
    :return: List of recommended content IDs
    """
    user = user_id or select_user(df)
    # Get all previous logs for the user
    user_authors, user_books = get_user_data(df, user)
    # Query book IDs by authors and drop all books the user already read
    recommendations = set(books_by_authors(df, user_authors)).difference(user_books)
    recommendations = list(recommendations)[:limit]
    display_recommendations(recommendations)
    return recommendations


def books_by_authors(df: pd.DataFrame, authors: list) -> list:
    """Function selects content IDs by authors names.
    :param df: DataFrame with all available data
    :param authors: List of authors names
    :return: List of content IDs in random order
    """
    df = df[df['Book-Author'].isin(authors)].drop_duplicates(subset=['Content_ID']).sample(frac=1.)
    recommendations = df['Content_ID'].to_list()
    return recommendations


def get_user_attributes(df: pd.DataFrame, user) -> tuple:
    """Function finds basic user attributes:
    indicated country of residence and age group.
    :param df: DataFrame with all available data
    :param user: User ID
    :return: Tuple with 2 values (country, age group)
    """
    country, age_group = df[df['User-ID'] == user].iloc[0, :][['Country', 'Age_group']]
    print(f'User region: {country}')
    print(f'User age group: {age_group}')
    return country, age_group


def recommend_by_user_attributes(df: pd.DataFrame, user_id=None, n_books=10) -> tuple:
    """Function produces generic recommendations
    based on user location and age group.
    :param df: DataFrame with all available data
    :param user_id: User ID
    :param n_books: Maximum number of books to recommend
    :return: Tuple with 2 lists of recommended content IDs (by readings, by ratings)
    """
    user = user_id or select_user(df)
    country, age = get_user_attributes(df, user)

    # Get generic recommendations based on user location and age group
    recommendations_1 = recommend_by_readings(
        df, countries=[country], age_groups=[age], n_books=n_books)
    recommendations_2 = recommend_by_ratings(
        df, countries=[country], age_groups=[age], n_books=n_books)

    return recommendations_1, recommendations_2


def recommend_by_title_similarity(df: pd.DataFrame,
                                  user_id=None,
                                  n_titles: int = 5,
                                  top_n: int = 10):
    """Function produces embeddings for book titles
    and performs semantic similarity search using a list
    of book titles passed as a query.
    :param df: DataFrame with all available data
    :param user_id: User ID
    :param n_titles: Number of titles to select from user history
    :param top_n: Number of similar titles to search for
    :return: Map object containing recommended book titles
    """
    global title_embeddings

    user = user_id or select_user(df)

    query = get_user_data(df, user, titles_only=True)
    print(f'Top-5 books with highest ratings:')
    query = query[:n_titles]
    for title in query:
        print(title)

    # All unique content IDs and titles
    df = df[['Content_ID', 'Book-Title']].drop_duplicates().reset_index(drop=True).copy()

    # Pretrained NLP model
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    # If the function was called for the 1st time,
    # we have to convert all titles into embeddings.
    if title_embeddings is None:
        title_embeddings = model.encode(df['Book-Title'].to_list(), convert_to_tensor=True)
        print(f'Book titles converted to embeddings. Shape: {title_embeddings.shape}')
        query_embeddings = model.encode(query, convert_to_tensor=True)
        print(f'Query titles converted to embeddings. Shape: {query_embeddings.shape}')

    # For repeated calls when embeddings are already available
    else:
        indexes = df[df['Book-Title'].isin(query)].index
        query_embeddings = title_embeddings[indexes]
        print(f'Embedding for query titles selected. Shape: {query_embeddings.shape}')

    # For every title in the original query extract indexes of similar titles
    recommendations = []
    result = util.semantic_search(query_embeddings, title_embeddings, top_k=top_n)
    for query in result:
        for similar in query:
            # Do not recommend identical titles
            # and title with low similarity score.
            if 0.75 < similar['score'] < 0.95:
                recommendations.append(similar['corpus_id'])

    # Replace indexes by respective titles
    recommendations = map(lambda x: df.loc[x, 'Book-Title'], recommendations)
    print('Semantically similar titles:')
    for title in recommendations:
        print(title)

    return recommendations


def vectorize_users(df: pd.DataFrame) -> pd.DataFrame:
    """Function produces users embeddings
    based on reading history, age and location.
    Readings of popular authors are used as preferences features.
    Age group and location are one-hot-encoded and added to preferences.
    :param df: DataFrame with all available data
    :return: DataFrame with users embeddings of shape n_users x n_features
    """
    # Select 2000 most popular authors
    popular_authors = df['Book-Author'].value_counts().nlargest(2000)
    popular_authors = set(popular_authors.index)
    print(f'Identified {len(popular_authors)} popular authors')

    # Group less popular authors in a separate category and create a pivot table
    df['Book-Author'] = df['Book-Author'].apply(lambda x: x if x in popular_authors else 'other')
    users = df.pivot_table(index='User-ID', columns='Book-Author',
                           values='Book-Rating', aggfunc='count')
    users = users.fillna(0)

    # To get a normalized representation of users' preferences
    # we divide number of readings per popular author
    # by the total number of reading by each user.
    shape = users.shape
    total = users.sum(axis=1)
    total_arr = np.repeat(total.values, shape[1]).reshape(shape)
    index = users.index
    columns = users.columns
    users = pd.DataFrame(users.values / total_arr, index=index, columns=columns)
    print(f'User embeddings based on authors extracted. Shape: {users.shape}')

    # Drop duplicate rows from the original DataFrame
    # and sort users in the same order as in embeddings array
    df = df.drop_duplicates(subset=['User-ID']).sort_values(by='User-ID').set_index('User-ID')
    # Encode age categories and countries
    users = pd.concat(
        (users, pd.get_dummies(df['Age_group']), pd.get_dummies(df['Country'])),
        axis='columns'
    )
    print(f'Age and location added. Shape: {users.shape}')

    return users


def select_unread_books(df: pd.DataFrame,
                        sim_users: np.array,
                        read_books: set,
                        limit: int = 100) -> list:
    """Function selects books that were read and were not
    explicitly disliked by 'sim_users', which are not present
    in 'read_books'.
    :param df: DataFrame with all available data
    :param sim_users: Array of IDs representing similar users
    :param read_books: Set of content IDs from the history of query user
    :param limit: Maximum number of books to select from similar users logs
    :return: List of recommended content IDs
    """
    # Select books read by similar users, except the books rated between 1 and 4
    # (obviously disliked)
    u_books = df[
        (df['User-ID'].isin(sim_users))
        & ((df['Book-Rating'] >= 5) | (df['Book-Rating'] == 0))
        ].copy()
    print(f'Selected {len(u_books)} logs except books rated 1 through 4')

    # Add a frequency column for read authors
    u_books['Readings_per_author'] = (
        u_books.groupby('User-ID')  # for each user
        ['Book-Author'].transform('count')  # count number of readings per each author
    )

    # Filter out books with 0 rating if the user read only one book of this author
    drop_idx = u_books[(u_books['Readings_per_author'] == 1) & (u_books['Book-Rating'] == 0)].index
    u_books.drop(drop_idx, inplace=True)
    print(f'Selected {len(u_books)} logs except 0-rated books with one reading per author')
    # Count readings per unique 'Content_ID'
    u_books = Counter(u_books['Content_ID'])
    n_items = len(u_books)
    print(f'Total number of unique content IDs: {n_items}')
    # Reduce total number of selected books if necessary
    if n_items > limit:
        u_books = pd.DataFrame({'Content_ID': u_books.keys(), 'n_users': u_books.values()})
        u_books.sort_values(by='n_users', ascending=False, inplace=True)
        u_books = u_books.head(limit)['Content_ID'].values
        print(f'Reduced the number of unique content IDs to {limit}')
    else:
        u_books = u_books.keys()

    # Select the books the query user did not read
    recommendations = list(set(u_books).difference(read_books))
    print(f'Total number of recommended content IDs after dropping read books: {len(recommendations)}')

    return recommendations


def recommend_by_similar_users(user_id: int,
                               df: pd.DataFrame,
                               embeddings: pd.DataFrame,
                               top_k: int = 11) -> list:
    """Function searches 'embeddings' for vectors most similar
    to the 'user_id', selects books that were read and were not
    explicitly disliked by similar users, which are not present
    in the 'user_id' history.
    :param user_id: Integer ID for the user
    :param df: DataFrame with all available data
    :param embeddings: DataFrame of embeddings for all users
    :param top_k: Maximum number of similar user vectors to search
    :return: List of recommended content IDs
    """
    # Array representing the query user
    query = embeddings.loc[user_id, :].values
    # Information about the query user
    authors, read_books = get_user_data(df, user_id)
    print(f'User {user_id} read authors:', authors)

    # Search for top_k vectors most similar to the query
    # (returns a pd.DataFrame with 2 columns: 'corpus_id' and 'score')
    similar_users = pd.DataFrame(
        util.semantic_search(query, embeddings.values, top_k=top_k)[0]
    )
    # Add users IDs finding them by row indexes
    similar_users['User-ID'] = embeddings.iloc[similar_users['corpus_id'], :].index
    print(f'Similar users:\n{similar_users}')

    # Drop 1st row which represents the user that was used as a query
    # (self-match with similarity score=1.0)
    similar_users = similar_users.iloc[1:, :]['User-ID']

    # Select books from similar users histories except disliked books,
    # one-off accidental readings and books present in current user's logs
    recommendations = select_unread_books(df, similar_users, read_books)
    display_recommendations(recommendations)

    return recommendations


def recommend_by_similar_users_faiss(index,
                                     user_id: int,
                                     df: pd.DataFrame,
                                     embeddings: pd.DataFrame,
                                     top_k: int = 11) -> list:
    """
    :param index: FAISS index of type IndexFlatL2
    :param user_id: Integer ID for the user
    :param df: DataFrame with all available data
    :param embeddings: DataFrame of embeddings for all users
    :param top_k: Maximum number of similar user vectors to search
    :return: List of recommended content IDs
    """
    # Information about the query user
    authors, read_books = get_user_data(df, user_id)
    print(f'User {user_id} read authors:', authors)

    # User vector
    query_vector = embeddings.loc[user_id, :].values.astype('float32').reshape(1, -1)
    # Search for 10 most similar indexes for the user's vector (+self-match)
    matched_emb, matched_indexes = index.search(query_vector, top_k)  # Row indexes of similar users
    # Get users IDs by row indexes
    similar_ids = embeddings.iloc[matched_indexes[0], :].index
    similar_ids = similar_ids[1:]  # Drop query user (self-match)
    print(f'Similar users:\n{similar_ids}')

    # Select books from similar users histories except disliked books,
    # one-off accidental readings and books present in current user's logs
    recommendations = select_unread_books(df, similar_ids, read_books)
    display_recommendations(recommendations)

    return recommendations
