"""Module for exploratory data analysis (EDA).
"""

import matplotlib.pyplot as plt
import recommender_utils as utils  # Utility functions

# Extract all data and combine into a single DataFrame
data = utils.get_data()

# Index(['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher',
#        'Image-URL-S', 'Image-URL-M', 'Image-URL-L', 'User-ID', 'Book-Rating',
#        'Location', 'Age', 'Country', 'Age_group'],
#       dtype='object')

# Content analysis
for column in ('ISBN', 'Book-Author', 'Publisher', 'Book-Title'):
    print('-' * 100)
    print(f'{column}: {data[column].nunique()} unique values')
    counts = data[column].value_counts().nlargest(20)
    plt.bar(counts.index, counts.values)
    plt.xticks(rotation=90)
    plt.title(f'{column} Most Frequent Values')
    plt.tight_layout()
    plt.show()
    print('Top-20 most frequent values:')
    print(counts)

# Distribution of publication year for unique ISBN
plt.hist(data.drop_duplicates(subset=['ISBN'])['Year-Of-Publication'],
         bins=[1900, 1950, 1960, 1970, 1980, 1990, 1995, 2000, 2005])
plt.title('Book Novelty: Year of Publication')
plt.show()

# Ratings are from 1 to 10. 0 means the user did not rate the book.
plt.hist(data['Book-Rating'], bins=30)
plt.title('Ratings Distribution')
plt.show()

# Number of ratings per user
plt.hist(data.groupby(by='User-ID')['Book-Rating'].count(),
         bins=30, log=True)
plt.title('Number of Ratings per User')
plt.show()

# Users by country
print(f'{data["Country"].nunique()} countries')
counts = data.drop_duplicates(subset=['User-ID'])['Country'].value_counts().nlargest(20)
plt.bar(counts.index, counts.values)
plt.xticks(rotation=90)
plt.title(f'Users Locations (Most Frequent)')
plt.tight_layout()
plt.show()

# Users by age group
counts = data.drop_duplicates(subset=['User-ID'])['Age_group'].value_counts()
groups = ['unknown', 'preschool', 'primary_school', 'secondary_school',
          'teenager', 'young_adult', 'adult', 'senior']
plt.bar(groups, counts[groups].values)
plt.xticks(rotation=90)
plt.title(f'Age Groups')
plt.tight_layout()
plt.show()
