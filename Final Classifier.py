
# coding: utf-8

# # Amazon and Rotten Tomatoes Entity Resolution
# ### Adam Coviensky (ac4092), Ian Johnson (icj2103)
# ### Instabase submission: icj2103

# Our entity resolution technique involved joining the Amazon and Rotten Tomatoes databases for the ID’s listed in the training set. We then matched the schemas for the features contained in both of the datasets and created new features from these. These features included the absolute value of the difference in time. Boolean features indicating whether or not the directors of the two movies are the same, the runtimes are the same, and whether or not the runtimes are the same within some range (we used 3 minutes). We then created features for the number of matching actors and for the percentage of matching actors. Finally, we performed cross-validation with a customized gridsearch using the training set to determine which features we should be keeping in our final model and which classifier to use between a decision tree, a random forest, and the gradient boosting classifier.
#  
# Our model had a precision, recall, and F1-scores of 0.97, 0.97 and 0.96 respectively for the training set. We obtained these scores by using train_test_split on our training set and fitting the model to the training portion and testing it on the testing portion.
#  
#  
# The most important features were determined to be ‘time_same_2’ which is a binary variable indicating whether the two runtimes are equivalent within a range of 3 minutes. Also, ‘directors_same’, indicated whether the two directors are the same.  ‘num_match_stars’ and ‘percent_match_stars’ indicated the number of matching stars and the percentage of the stars matched from the Amazon dataset respectively. These were the features we used to train our final model.
#  
# We avoided pairwise comparison of all movies across both datasets by simply joining the id’s listed in the training set to create a model. We trained a random forest classifier on this training data with our newly created features. This allowed us to predict whether two new movies matched using our trained random forest on a new combination of two movies.

# In[18]:

import re
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

ama = pd.read_csv('./data/amazon.csv')
rt = pd.read_csv('./data/rotten_tomatoes.csv')

holdout = pd.read_csv('./holdout.csv')
test = pd.read_csv('./test.csv')
train = pd.read_csv('./train.csv')


# ### Define Columns

# In[19]:

ama.columns = [
    'id_left',
    'time_left',
    'director_left',
    'star_left',
    'cost_left'
]

rt.columns = [
    'id_right',
    'time_right',
    'director_right',
    'year_right',
    'star1_right',
    'star2_right',
    'star3_right',
    'star4_right',
    'star5_right',
    'star6_right',
    'rotten_tomatoes_right',
    'audience_rating_right',
    'review1_right',
    'review2_right',
    'review3_right',
    'review4_right',
    'review5_right'
]


# ### Feature Creation Function

# In[20]:

def create_features(dataset):
    new_data = pd.DataFrame(dtype=str)

    for index, row in dataset.iterrows():
        amarow = ama[ama['id_left'] == row['id1']]
        rtrow = rt[rt['id_right'] == row['id 2']]

        amarow.reset_index(drop=True, inplace=True)
        rtrow.reset_index(drop=True, inplace=True)

        new_row = pd.concat([amarow, rtrow], axis=1)
        new_data = pd.concat([new_data, new_row])

    new_data.fillna('0', inplace = True)
    new_data.dropna()

    # Compute directors match column
    new_data['directors_same'] = (new_data['director_left'] == new_data['director_right']).astype(int)

    # Compute time columns
    new_data['time_left'] = new_data['time_left'].astype(str)
    new_data['time_right'] = new_data['time_right'].astype(str)
    new_data['time_norm_left'] = new_data['time_left'].apply(compute_time_norm)
    new_data['time_norm_right'] = new_data['time_right'].apply(compute_time_norm)
    new_data['time_same'] = (new_data['time_norm_left'].astype(int) == new_data['time_norm_right'].astype(int)).astype(int)
    new_data['time_same_2'] = (abs((new_data['time_norm_left'].astype(int) - new_data['time_norm_right'].astype(int)).astype(int)) <= 3).astype(int)
    new_data['time_diff'] = (new_data['time_norm_left'].astype(int) - new_data['time_norm_right'].astype(int)).astype(int)
    new_data['time_diff_abs'] = abs(new_data['time_diff']).astype(int)

    # Compute actors columns
    actors_split = new_data['star_left'].str.split(', ', expand=True)
    for i in range(6 - actors_split.shape[1]):
        actors_split[str(i)] = ""
    actors_split.columns = ['star_' + str(i) for i in range(6)]
    new_data = pd.concat([new_data, actors_split], axis=1)

    # Number actors match
    cols = list(new_data.loc[:,'star_0':'star_4']) + list(new_data.loc[:,'star1_right':'star6_right'])
    new_data['num_match_stars'] = new_data[cols].apply(compute_number_stars_match, axis = 1)

    # Percent actors match
    new_data['percent_match_stars'] = new_data[cols].apply(compute_percent_stars_match, axis = 1)

    return new_data


# ### Helper Feature Creation Functions

# In[21]:

def compute_number_stars_match(row):
    actors_left = ['star_0', 'star_1', 'star_2', 'star_3', 'star_4']
    actors_right = ['star1_right', 'star2_right', 'star3_right', 'star4_right', 'star5_right', 'star6_right']
    list_left = row.loc[actors_left].tolist()
    list_right = row.loc[actors_right].tolist()

    # Avoid matching nulls
    list_left = filter(None, list_left)
    list_right = filter(None, list_right)

    return len(np.intersect1d(list_left, list_right))


def compute_percent_stars_match(row):
    actors_left = ['star_0', 'star_1', 'star_2', 'star_3', 'star_4']
    actors_right = ['star1_right', 'star2_right', 'star3_right', 'star4_right', 'star5_right', 'star6_right']
    list_left = row.loc[actors_left].tolist()
    list_left = filter(None, list_left)
    list_right = row.loc[actors_right].tolist()
    x = float(len(np.intersect1d(list_left, list_right)))
    ama_num = len(list_left)
    return x / ama_num


regex = re.compile(r'[0-9]*')

def compute_time_norm(row):
    row = str(row)

    if '/' in row:
        # Invalid time entry
        print(row)
        return 0

    match = regex.findall(row)
    temp = filter(None, match)
    if len(temp) == 2:
        return 60*int(temp[0]) + int(temp[1])
    if len(temp) == 1:
        return temp[0]


def remove_bad_samples(data):

    train.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data = pd.concat([data, train], axis=1)

    # Remove bad training rows
    data = data[data['id_left'] != 199]
    data = data[data['id_left'] != 680]
    data = data[data['id_left'] != 487]
    data = data[data['id_left'] != 756]
    data = data[data['id_left'] != 770]
    data = data[data['id_left'] != 1701]

    return data


# ### Predict With Training Data

# In[23]:

features_cols = [
    'time_same_2',
    'directors_same',
    'num_match_stars',
    'percent_match_stars',
]

train_data = create_features(train)
train_data = remove_bad_samples(train_data)

x_train, x_test, y_train, y_test = train_test_split(train_data[features_cols], train_data['gold'], random_state=10)

clf = RandomForestClassifier(random_state=42)
clf.fit(x_train, y_train)
scores = clf.score(x_test, y_test)

preds = clf.predict(x_test)

# F1 score:
from sklearn.metrics import classification_report
print(classification_report(y_test, preds))


# ### Final Predictions

# In[24]:

features_cols = [
    'time_same_2',
    'directors_same',
    'num_match_stars',
    'percent_match_stars',
]


# Test Predictions
train_data = create_features(train)
train_data = remove_bad_samples(train_data)
test_data = create_features(test)

clf = RandomForestClassifier()
clf.fit(train_data[features_cols], train_data['gold'])
test_preds = clf.predict(test_data[features_cols])

test_preds = pd.DataFrame(test_preds)
test_preds.columns = ['gold']
test_preds.to_csv('test_gold.csv', index=False)


# Holdout Predictions
train_data = create_features(train)
train_data = remove_bad_samples(train_data)
holdout_data = create_features(holdout)

clf = RandomForestClassifier()
clf.fit(train_data[features_cols], train_data['gold'])
holdout_preds = clf.predict(holdout_data[features_cols])

holdout_preds = pd.DataFrame(holdout_preds)
holdout_preds.columns = ['gold']
holdout_preds.to_csv('holdout_gold.csv', index=False)


# In[ ]:



