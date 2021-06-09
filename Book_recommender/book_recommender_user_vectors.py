"""Module for recommendation algorithm based on vectorization of users
and querying reading history of similar users.
Algorithm steps:
1. Transform information about the users into 1D vectors:
   - Count the number of times each user read most popular authors
     (2000 most popular authors are counted, all other authors are counted together
     in a joint feature-group)
   - Normalize this data by total number of readings by each user to get a distribution
     for each user (vectors contain values between 0 and 1).
   - Add one-hot-encoded data on user age group ('unknown', 'preschool', 'primary_school',
     'secondary_school', 'teenager', 'young_adult', 'adult', 'senior') and location (country)
2. Search for similar users based on cosine similarity of vectors.
3. Select books that similar users read and did not explicitly disliked.
4. Discard accidental items (books with 0 ratings where similar user read the author only once).
5. To limit total number of recommendations sort all previously selected books
   by the number of readings by similar users and select top-100.
6. Drop the books that the user in question had already read.
This algorithm produces a version of collaborative filtering recommendations.
Advantages:
- Interpretability: selection rules are explicitly defined and could be fine-tuned
- Speed of search: in production environment vectors matrix for all users could be
  updated once a day or once a week and stored in a database or in cache.
  For larger databases faiss library could be used to index user vectors and search
  faster using CPU or GPU.
Disadvantages:
- Poor quality of recommendations for users with limited reading history.
  New users or users with preferences not matching popular list of authors
  could get irrelevant recommendations.
- Limited information about users in the original data and small number of meaningful attributes.
"""

import faiss
import recommender_utils as utils  # Utility functions

# Extract all data and combine into a single DataFrame
data = utils.get_data()

# Original data about the users is limited to age and location.
# We generate preferences vector for each user based on activity logs
# (which authors users read among the most popular 2000 authors)
# and combine distribution of readings with one-hot-encoded information
# about user age and location.

# Vector representation for all users
users_vectors = utils.vectorize_users(data)

# Select random user
user = utils.select_user(data)

# Select 10 most similar users (search for 11 most similar vectors
# because the result includes the user in question - self match)
result = utils.recommend_by_similar_users(user, data, users_vectors, top_k=11)

# Recommendations for other random user
user = utils.select_user(data)
result = utils.recommend_by_similar_users(user, data, users_vectors, top_k=11)

# ----------------- Faster index search using faiss library ----------------------

# Create FAISS index from all user embeddings
n_dimensions = users_vectors.shape[1]
fast_index = faiss.IndexFlatL2(n_dimensions)
fast_index.add(users_vectors.values.astype('float32', order='C'))

# Select 10 most similar users (default top_k)
result = utils.recommend_by_similar_users_faiss(fast_index, user, data, users_vectors)

# Select 3 most similar users (+self-match)
result = utils.recommend_by_similar_users_faiss(fast_index, user, data, users_vectors, top_k=4)
