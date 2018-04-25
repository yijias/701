import numpy as np
import math
import os
import scipy
import csv
import re
import pandas as pd
from re import search
import string
import statistics
#from sklearn.model_selection import train_test_split
#from sklearn import datasets

data_dir=os.path.join('..','RSdata') 

#######################Pull out movie id##########################
with open(os.path.join(data_dir,'movies.dat'),'r') as f:
	dmovie=pd.DataFrame(l.rstrip().split() for l in f)
# Load movie data into workspace
#print(dmovie.shape[0])
movie=["" for x in range(dmovie.shape[0])]
movieid=np.zeros(dmovie.shape[0])
genres=["" for x in range(dmovie.shape[0])]
#print(movie_id.shape)

# Movie data : movie (in syntax of '3201::Action|Thriller'),
# movieid (integer), movie genres (string).
for i in range(dmovie.shape[0]):
    movie[i]=dmovie.get_value(i,0)
        #print(movie[i])
    movieid[i]=int(re.search(r'\d+',movie[i]).group())
    genres[i]=movie[i].split('::')[1]
#print(movieid)
movie_n = dmovie.shape[0]
        
######################Pull out user id#########################
with open(os.path.join(data_dir,'users.dat'),'r') as f:
	duser=pd.DataFrame(l.rstrip().split() for l in f)
# Load user data into workspace
user=["" for x in range(duser.shape[0])]
userid=np.zeros(duser.shape[0])
for i in range(duser.shape[0]):	
	user[i]=duser.get_value(i,0,takeable=False)
	u=int(re.search(r'\d+',user[i]).group())
	userid[i]=u
#print(userid)
#print(userid)
# User data: userid
user_n = duser.shape[0]

##################Pull out training data, rating ,userid, movie id#####
with open(os.path.join(data_dir,'training_rating.dat'),'r') as f:
	drating=pd.DataFrame(l.rstrip().split() for l in f)

# Now we load Traning data as a whole into our workspace
train_user_id=np.zeros(drating.shape[0])  # user id
train_movie_id=np.zeros(drating.shape[0])  # profile id
train_rating=np.zeros(drating.shape[0])   # rating

for i in range(drating.shape[0]):
	training_data=drating.get_value(i,0)
	t=map(int,re.findall(r'\d+',training_data))
	#print(t)
	if len(t)==3:
		#print(t)
		train_user_id[i]=t[0]
		train_movie_id[i]=t[1]
		train_rating[i]=t[2]
	#print(train_user_id,train_movie_id,train_rating)


user_id=train_user_id[np.nonzero(np.asarray(train_user_id))][0:400000]
item_id=train_movie_id[np.nonzero(np.asarray(train_movie_id))][0:400000]
rating=train_rating[np.nonzero(np.asarray(train_rating))][0:400000]

# In total 899792 observed cases.

# Here I want to build the training R matrix
# There are in total 900102 rating observations.
# Instead of filling in missing data, we simply wouldn't change the missing values.(stay 0) 


#####################Matrix Factorization ALS#############################
#bi, bias of user
#cj, bias of movie

K=5
M=userid.shape[0]
N=movieid.shape[0]
mu=sum(rating)/rating.shape[0]
reg=0.1
#regMovie=0
U = np.random.randn(M, K)
V = np.random.randn(K, N)
Q=np.zeros([M,N])
B = np.zeros(M)
C = np.zeros(N)
mu=sum(rating)/rating.shape[0] 
#### creating dictionary for the set of all user i that have rated movie j with rating r.
ratings_by_i={}
for i in userid:
	pair=np.argwhere(user_id==i)
	ratings_by_i[i]=np.asarray([[item_id[pair][j],rating[pair][j]] for j in range(pair.shape[0])])
#### creating dictionary for the set of all movie j that have usr i have rated with rating r
ratings_by_j={}
for j in movieid:
	pair=np.argwhere(item_id==j)
	ratings_by_j[j]=np.asarray([[user_id[pair][k],rating[pair][k]] for k in range(pair.shape[0])])

Q=np.zeros([M,N])
for i in ratings_by_i:
	i=int(i)
	if ratings_by_i[i].shape[0]>0:
		ind_movie=ratings_by_i[i][:,0]
		ind_rating=ratings_by_i[i][:,1]
		ind_movie=[int(ind_movie[j]-1) for j in range(ind_movie.shape[0])]
		#print(ind_movie)
		Q[i-1,ind_movie]=ind_rating.ravel()


weighted_errors=[]
r_hat=np.zeros([M,N])
for t in range(20):
    # update B & U
	for i in range(M):
		if ratings_by_i[i+1].shape[0]>0:
			accum = 0
			for j, r in ratings_by_i[i+1]:
				j=int(j[0])
				r=int(r[0])
				accum+=(r-U[i,:].dot(V[:,j-1])-C[j-1])
				#print(ratings_by_i[i+1].shape[0])
			B[i]=accum/(1+reg)/ratings_by_i[i+1].shape[0]
	for i in range(M):
		if ratings_by_i[i+1].shape[0]>0:
			matrix = np.zeros((K, K)) + reg*np.eye(K)
			vector = np.zeros(K)
			for j, r in ratings_by_i[i+1]:
				j=int(j[0])
				r=int(r[0])
				matrix += np.outer(V[:,j-1], V[:,j-1])
				vector += (r - B[i] - C[j-1])*V[:,j-1]
			U[i,:] = np.linalg.solve(matrix, vector)
			ind1=np.argwhere(U[i,:]<0.0)
			U[i,ind1]=-1*U[i,ind1]
    # update C & V
	for j in range(N):
		if ratings_by_j[j+1].shape[0]>0:
			accum = 0
            #print(j)
			for i, r in ratings_by_j[j+1]:
				i=int(i[0])
				r=int(r[0])
				accum += (r - U[i-1,:].dot(V[:,j]) - B[i-1])
			C[j]=accum/(1 + reg)/(ratings_by_j[j+1].shape[0])
	for j in range(N):
		if ratings_by_j[j+1].shape[0]>0:
			matrix = np.zeros((K, K)) + reg*np.eye(K)
			vector = np.zeros(K)
			for i, r in ratings_by_j[j+1]:
				i=int(i[0])
				r=int(r[0])
				matrix += np.outer(U[i-1,:], U[i-1,:])
				vector += (r - B[i-1] - C[j])*U[i-1,:]
			V[:,j] = np.linalg.solve(matrix, vector)
			ind2=np.argwhere(V[:,j]<=0.0)
			V[ind2,j]=-1*V[ind2,j]
	r_hat=U.dot(V)

    # Prediction & RMSE
test