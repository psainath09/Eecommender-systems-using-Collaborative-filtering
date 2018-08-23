# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 12:52:14 2018

@author: satyasainath pulaparthi
"""
# import required packages
import numpy as np
import pandas as pd
from surprise import Reader, Dataset

# import data and preprocessing
data= pd.read_csv('recommender_data.csv')
data= data.drop(["user_sequence"],axis=1)
data["course"]= data["course_id"].apply(lambda x: x[2:])
data= data.drop(["course_id"],axis=1)
data["course"]= pd.to_numeric(data["course"])
data["completed"]=1
data= data.drop("challenge_sequence",axis=1)
data=data.assign(id=(data["course"]).astype('category').cat.codes)
preference_dict = {'taskID': list(data.course),'userID': list(data.learner_id),'task': list(data.completed)}
df = pd.DataFrame(preference_dict)
# read the dataset into scale reader for surprise
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(df[['userID', 'taskID', 'task']],reader)
#split the dataset into train and test
from surprise.model_selection import train_test_split 
train,test=train_test_split(data,test_size=0.25)
#import surprise packages
from surprise import SVD, evaluate,accuracy
from surprise.model_selection import cross_validate
from collections import defaultdict
# create an object for PMF and run it
# PMF and its rmse, mae results
pmf_results=[]
pmf_algo = SVD(n_factors=100,n_epochs=6,biased=False)
pmf_algo.fit(train)
pmf_predictions= pmf_algo.test(test)
pmf_results.append(accuracy.mae(pmf_predictions))
pmf_results.append(accuracy.rmse(pmf_predictions,verbose=True))

#This code has been taken from surprise package documentation
def precision_recall_at_k(predictions, k, threshold):
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Number of relevant items
        n_rel = sum((true_r == 1) for (_, true_r) in user_ratings)
        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r == 1) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k/n_rec_k  if n_rec_k >= k else 0
        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel >= k else 0
    return precisions,recalls

# see precision and recall values
precisions, recalls = precision_recall_at_k(pmf_predictions, k=3, threshold=0.5)
print(sum(prec for prec in precisions.values()) / len(precisions))
print(sum(rec for rec in recalls.values()) / len(recalls))

pmf_precision=[]
pmf_precision.append(sum(prec for prec in precisions.values()) / len(precisions))
pmf_precision.append(sum(rec for rec in recalls.values()) / len(recalls))
