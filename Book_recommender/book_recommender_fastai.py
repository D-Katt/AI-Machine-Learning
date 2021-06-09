"""Module for collaborative filtering based on fastai.collab library.
Model predicts book ratings in the range between 1 and 10.
Samples with 0 ratings where user did not rate the book
are not used in training and testing.
"""

from fastai.collab import CollabDataLoaders, collab_learner
import recommender_utils as utils  # Utility functions

# Extract all data and combine into a single DataFrame
data = utils.get_data()

# Absence of a rating does not mean that the user did not like the book.
# Actual rating values vary between 1 and 10. If we leave 0 ratings unchanged,
# the model could implicitly assume that 0 ratings are worse than the lowest actual rating.
# 0 ratings prevail in the dataset and outnumber all other rating values.
# Assigning average or median values to the majority of samples will distort the data
# and lead to inaccurate and unreliable recommendation model.
# To avoid this we will drop all rows with 0 values before passing the data to collaborative model.
data_col = data[data['Book-Rating'] > 0]
print(f'Number of samples before dropping 0 values: {len(data)}\n'
      f'Number of samples after dropping 0 values: {len(data_col)}')

# Prepare data for the model
dls = CollabDataLoaders.from_df(data_col[['User-ID', 'Content_ID', 'Book-Rating']],
                                bs=64, valid_pct=0.1)

# Create a collaborative model and train for 5 epochs
learner = collab_learner(dls, y_range=(1, 10))
learner.fine_tune(5)

# After 5 epochs of training validation loss stalls. If training continues,
# the model just overfits on the train data without any considerable improvement on the validation set.
# Validation error is about 3.7, which is high taking into account that ratings vary between 1 and 10.
# This magnitude of errors could easily result in recommending the books that the user wouldn't like
# and not recommending books that in reality would get high ratings.

# Display predicted and actual ratings for each user-book pair
learner.show_results()

# Prediction for all users
result = learner.predict(dls)
