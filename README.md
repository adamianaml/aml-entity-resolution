# aml-entity-resolution

Our entity resolution technique involved joining the Amazon and Rotten Tomatoes databases for the ID’s listed in the training set. We then matched the schemas for the features contained in both of the datasets and created new features from these. These features included the absolute value of the difference in time. Boolean features indicating whether or not the directors of the two movies are the same, the runtimes are the same, and whether or not the runtimes are the same within some range (we used 3 minutes). We then created features for the number of matching actors and for the percentage of matching actors. Finally, we performed cross-validation with a customized gridsearch using the training set to determine which features we should be keeping in our final model and which classifier to use between a decision tree, a random forest, and the gradient boosting classifier.
 
Our model had a precision, recall, and F1-scores of 0.97, 0.97 and 0.96 respectively for the training set. We obtained these scores by using train_test_split on our training set and fitting the model to the training portion and testing it on the testing portion.
 
 
The most important features were determined to be ‘time_same_2’ which is a binary variable indicating whether the two runtimes are equivalent within a range of 3 minutes. Also, ‘directors_same’, indicated whether the two directors are the same.  ‘num_match_stars’ and ‘percent_match_stars’ indicated the number of matching stars and the percentage of the stars matched from the Amazon dataset respectively. These were the features we used to train our final model.
 
We avoided pairwise comparison of all movies across both datasets by simply joining the id’s listed in the training set to create a model. We trained a random forest classifier on this training data with our newly created features. This allowed us to predict whether two new movies matched using our trained random forest on a new combination of two movies.



# todo
- edit distance
- time +/- a few mins
  - log or ln time diff
- external data
  - completing movie names...?
- remove misclassified points
