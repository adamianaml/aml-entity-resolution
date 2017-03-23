
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[ ]:

ama = pd.read_csv('./data/amazon.csv')
rt = pd.read_csv('./data/rotten_tomatoes.csv')

holdout = pd.read_csv('./data/holdout.csv')
test = pd.read_csv('./data/test.csv')
train = pd.read_csv('./data/train.csv')


# In[ ]:

ama.head()
a = pd.DataFrame(ama[ama['id'] == 4])
ama.columns = [
    'id_left',
    'time_left',
    'director_left',
    'star_left',
    'cost_left'
]


# In[ ]:

b = pd.DataFrame(rt[rt['id'] == 3668])
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


# In[ ]:

new_data = pd.DataFrame(dtype=str)


# In[ ]:

# Merge records from ama and rt
for index, row in train.iterrows():
    amarow = ama[ama['id_left'] == row['id1']]
    rtrow = rt[rt['id_right'] == row['id 2']]
    gold = pd.DataFrame(data={'gold': [row['gold']]})
    
    amarow.reset_index(drop=True, inplace=True)
    rtrow.reset_index(drop=True, inplace=True)
    
    new_row = pd.concat([amarow, rtrow, gold], axis=1)
    new_data = pd.concat([new_data, new_row])


# In[ ]:

# Same director column
new_data['directors_same'] = (new_data['director_left'] == new_data['director_right']).astype(int)

# Split actors into unique columns
actors_split = new_data['star_left'].str.split(',', expand=True)
actors_split_columns = ['star_' + str(i) for i in range(len(actors_split.columns))]
actors_split.columns = actors_split_columns
new_data = pd.concat([new_data, actors_split], axis=1)

star_columns = [
    'star1_right',
    'star2_right',
    'star3_right',
    'star4_right',
    'star5_right',
    'star6_right',
]

# Compute common actor count column
new_data['common_actor_count'] = pd.Series()

for i, row in new_data.iterrows():
    actors_left = set(row[actors_split_columns])
    actors_left.discard(None)
    actors_right = set(row[star_columns])
    actors_right.discard(None)
    
    new_data.iloc[i, new_data.columns.get_loc('common_actor_count')] = len(actors_left.intersection(actors_right))
#     print(new_data.iloc[i, new_data.columns.get_loc('common_actor_count')])


# In[ ]:

pd.options.display.max_columns = 999
new_data['common_actor_count']


# In[ ]:

features = []

for i, row in train.iterrows():
    a = ama[ama['id_left'] == row['id1']]
    b = rtrow[rtrow['id_right'] == row['id 2']]
    features.append()

[ 1 for i, row in train.iterrows() if ama[ama['id_left'] == row['id1'] == rtrow[rtrow['id_right'] == row['id 2']] else 0 ]


# In[ ]:

clf = RandomForestClassifier()
x_train, x_test, y_train, y_test = train_test_split(new_data.ix[:, new_data.columns != 'gold'], new_data.iloc[:,-1])
clf.fit(x_train, y_train)
clf.score(x_test, y_test)


# In[ ]:



