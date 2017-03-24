
# coding: utf-8

# In[843]:

import re
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

pd.options.display.max_columns = 999


# ### Load Data

# In[844]:

ama = pd.read_csv('./data/amazon.csv')
rt = pd.read_csv('./data/rotten_tomatoes.csv')

holdout = pd.read_csv('./data/holdout.csv')
test = pd.read_csv('./data/test.csv')
train = pd.read_csv('./data/train.csv')


# In[845]:

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


# ### Merge records from ama and rt

# In[861]:

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
    new_data['time_diff'] = (new_data['time_norm_left'].astype(int) - new_data['time_norm_right'].astype(int)).astype(int)
    
    # Compute actors columns
    actors_split = new_data['star_left'].str.split(', ', expand=True)
    actors_split_columns = ['star_' + str(i) for i in range(len(actors_split.columns))]
    actors_split.columns = actors_split_columns
    new_data = pd.concat([new_data, actors_split], axis=1)
    
    # Todo - fix for > 2 stars_left
    cols = list(new_data.loc[:,'star_0':'star_4']) + list(new_data.loc[:,'star1_right':'star6_right'])
    new_data['num_match_stars'] = new_data[cols].apply(compute_number_stars_match, axis = 1)
    
    return new_data


# In[862]:

train_data_copy = create_features(train)

actors_split = train_data_copy['star_left'].str.split(', ', expand=True)
actors_split_columns = ['star_' + str(i) for i in range(len(actors_split.columns))]
actors_split.columns = actors_split_columns
train_data_copy = pd.concat([train_data_copy, actors_split], axis=1)


# In[863]:

def compute_number_stars_match(row):
    actors_left = ['star_0', 'star_1', 'star_2', 'star_3', 'star_4']
    actors_right = ['star1_right', 'star2_right', 'star3_right', 'star4_right', 'star5_right', 'star6_right']
    list_left = row.loc[actors_left].tolist()
    list_right = row.loc[actors_right].tolist()
    x = len(np.intersect1d(list_left, list_right))
    
    return x


# In[864]:

regex = re.compile(r'[0-9]*')

def compute_time_norm(row):
    row = str(row)
    match = regex.findall(row)
    temp = filter(None, match)
    if len(temp) == 2:
        return 60*int(temp[0]) + int(temp[1])
    if len(temp) == 1:
        return temp[0]


# ### Predict with train

# In[850]:

features_cols = ['time_diff', 'directors_same', 'num_match_stars']

train_data = create_features(train)

train.reset_index(drop=True, inplace=True)
train_data.reset_index(drop=True, inplace=True)

train_data = pd.concat([train_data, train], axis=1) #, how='inner')

# Remove bad training rows
train_data = train_data[train_data['id_left'] != 199]
train_data = train_data[train_data['id_left'] != 680]

x_train, x_test, y_train, y_test = train_test_split(train_data[features_cols], train_data['gold'])

clf = GradientBoostingClassifier(min_samples_split=10)
clf.fit(x_train, y_train)
clf.score(x_test, y_test)


# In[857]:




# In[855]:

clf = GradientBoostingClassifier(n_estimators=200)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)

# Missed preds:
errors = preds - y_test
error_ind = np.nonzero(errors)[0]

print(preds[error_ind])
print(y_test.as_matrix()[error_ind])

rel_cols = ['director_left', 'time_right', 'directors_same', 'time_same', 'time_diff', 'num_match_stars', 'gold']
x_test.iloc[error_ind]

# from sklearn.metrics import confusion_matrix
# confusion_matrix(y_test, preds)


# ### Predict with test

# In[768]:

features_cols = ['time_same'] + ['directors_same'] + ['num_match_stars']
test_data = create_features(test)

clf = GradientBoostingClassifier(min_samples_split=10)
clf.fit(train_data[features_cols], train_data['gold'])
test_preds = clf.predict(test_data[features_cols])


# In[769]:

test_preds = pd.DataFrame(test_preds)
test_preds.columns = ['gold']
test_preds.to_csv('test_gold.csv', index=False)


# ### Predict with Holdout

# In[770]:

features_cols = ['time_same'] + ['directors_same'] + ['num_match_stars']
holdout_data = create_features(holdout)

clf = GradientBoostingClassifier(min_samples_split=10)
clf.fit(train_data[features_cols], train_data['gold'])
holdout_preds = clf.predict(holdout_data[features_cols])


# In[771]:

holdout_preds = pd.DataFrame(holdout_preds)
holdout_preds.columns = ['gold']
holdout_preds.to_csv('holdout_gold.csv', index=False)


# In[ ]:



