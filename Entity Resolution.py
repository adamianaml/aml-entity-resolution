
# coding: utf-8

# In[235]:

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[248]:

ama = pd.read_csv('./data/amazon.csv')
rt = pd.read_csv('./data/rotten_tomatoes.csv')

holdout = pd.read_csv('./data/holdout.csv')
test = pd.read_csv('./data/test.csv')
train = pd.read_csv('./data/train.csv')


# In[256]:

train


# In[222]:

ama.head()
a = pd.DataFrame(ama[ama['id'] == 4])
ama.columns = [
    'id_left',
    'time_left',
    'director_left',
    'star_left',
    'cost_left'
]


# In[221]:

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


# In[225]:

new_data = pd.DataFrame(dtype=str)


# In[227]:

# Merge records from ama and rt
for index, row in train.iterrows():
    amarow = ama[ama['id_left'] == row['id1']]
    rtrow = rt[rt['id_right'] == row['id 2']]
    gold = pd.DataFrame(data={'gold': [row['gold']]})
    
    amarow.reset_index(drop=True, inplace=True)
    rtrow.reset_index(drop=True, inplace=True)
    
    new_row = pd.concat([amarow, rtrow, gold], axis=1)
    new_data = pd.concat([new_data, new_row])


# In[246]:

new_data['directors_same'] = (new_data['director_left'] == new_data['director_right']).astype(int)

actors_split = new_data['star_left'].str.split(',', expand=True)
actors_split_columns = ['star_' + str(i) for i in range(len(actors_split.columns))]
actors_split.columns = actors_split_columns
new_data = pd.concat([new_data, actors_split], axis=1)



new_data['same_actor_count'] = len()


# In[ ]:

features = []

for i, row in train.iterrows():
    a = ama[ama['id_left'] == row['id1']]
    b = rtrow[rtrow['id_right'] == row['id 2']]
    features.append()

[ 1 for i, row in train.iterrows() if ama[ama['id_left'] == row['id1'] == rtrow[rtrow['id_right'] == row['id 2']] else 0 ]


# In[234]:

clf = RandomForestClassifier()
x_train, x_test, y_train, y_test = train_test_split(new_data.ix[:, new_data.columns != 'gold'], new_data.iloc[:,-1])
clf.fit(x_train, y_train)
clf.score(x_test, y_test)


# In[ ]:



