"""Module for content based recommendation algorithm.
Based on the user history the algorithm selects most highly rated books
and searches for semantically similar titles using pretrained NLP model
(Distil RoBERTa from sentence_transformers library).
If the original data contained features like book genre, language or text length,
it would be possible to produce more meaningful recommendations based on content clustering.
Disadvantages:
- Algorithm often matches identical books with minor differences in titles
  (punctuation or description of cover type added to the title in parenthesis).
- Depending on the degree of text similarity results can be irrelevant and annoying to the user.
- Algorithm does not take into account that for some types of specialized content
  (like language dictionaries) user who already obtained one item does not need
  similar items of the same type.
"""

import recommender_utils as utils  # Utility functions

# Extract all data and combine into a single DataFrame
data = utils.get_data()

# Available data does not include categories like book genre, language,
# text length, cover type, etc., which limits out ability to cluster content.
# We have information about author, title, year of publication and publisher.
# Using NLP models we can vectorize titles and search for similar titles.
result = utils.recommend_by_title_similarity(data)

# Recommendations for other random user
result = utils.recommend_by_title_similarity(data)
