"""Module for book recommendation algorithms based on majority vote,
user history and user attributes.
Generic recommendations algorithms:
- By book popularity: recommend books with largest number of readings
- By book ratings: recommend books with highest average ratings
Generic recommendations augmented by user attributes:
- Cohorts based approach: identify user location and age group
  and obtain generic recommendation for this geography and demographics
Generic and cohorts based algorithms could be marked as "popular" or "trending"
to inform the user that they are not actually personalized.
Recommendations based on user history:
- Query by author: identify authors that the user previously read and did not
  explicitly disliked and recommend other books by the same authors
This rules based approach takes into account user behaviour and is the most personalized.
"""

import recommender_utils as utils  # Utility functions

# Extract all data and combine into a single DataFrame
data = utils.get_data()

# ------------------- Generic recommendation algorithms -------------------------

# Implementation of recommendation algorithms that are not user specific
# or personalized and are based on a majority vote.

# 10 most popular books (books with the largest readership base)
result = utils.recommend_by_readings(data)

# 10 most popular books in Germany
result = utils.recommend_by_readings(data, countries=['germany'])

# 50 most popular books
result = utils.recommend_by_readings(data, n_books=50)

# 10 most popular books for senior age group
result = utils.recommend_by_readings(data, age_groups=['senior'])

# Recommendation by highest average book ratings.
# Default settings: 10 books with at least 10 ratings.
result = utils.recommend_by_ratings(data)

# 10 highest rated books with at least 100 ratings
result = utils.recommend_by_ratings(data, threshold=100)

# When both threshold and required number of recommendations are set too high,
# the function returns as many books as it can find.
# Query for 30 highest rated books with at least 500 ratings.
result = utils.recommend_by_ratings(data, n_books=30, threshold=500)

# Top-10 most highly rated books in Spain and Italy
result = utils.recommend_by_ratings(data, countries=['spain', 'italy'])

# Top-10 most highly rated books for teenagers and school age children
result = utils.recommend_by_ratings(
    data, age_groups=['primary_school', 'secondary_school', 'teenager']
)

# Top-10 most highly rated books for 20-35 year-olds in France
result = utils.recommend_by_ratings(
    data, age_groups=['young_adult'], countries=['france']
)

# ------------- Recommendation algorithms based on user attributes ------------------

# Implementation of cohorts based approach: we identify user location and age group
# and produce generic recommendations (majority vote) based on these attributes.

# 1st list is based on the total readings in the same geography and age group.
# 2nd list is based on average ratings in the same geography and age group.
res_1, res_2 = utils.recommend_by_user_attributes(data)

# Recommendations for other random user
res_1, res_2 = utils.recommend_by_user_attributes(data)

# ---------------- Recommendation algorithm based on user activity -----------------------

# Implementation of rules based approach: based on user activity
# (which authors user read and what ratings posted) we selects authors
# that the user read and did not explicitly disliked and recommend
# remaining books by the same authors that the user did not read yet.

# Select random user and produce recommendations
# based user readings history
result = utils.recommend_by_activity(data)

# Recommendations for other random user
result = utils.recommend_by_activity(data)
