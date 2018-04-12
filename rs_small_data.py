import os
import numpy as np
import random
import statistics
def findValid():
	#The raw list of all reviews
    path = os.getcwd()
    file = path + '/ratings_Musical_Instruments.csv'
    f = open(file,'r')
    rawString = f.read()
    rawList = rawString.split('\n')[:-1]
    for i in range(len(rawList)):
        rawList[i] = rawList[i].split(',')
        rawList[i] = tuple(rawList[i][:-1])

    #Take the first column of rawList (reviewer ID strings)
    reviewerbyreviews = np.array(rawList)[:,0]
    #reviewers: reviewer list without repeat(string); counts: the number of reviews by each person
    reviewers, counts = np.unique(reviewerbyreviews, return_counts=True)
    #{reviewerID(string): pos number of the reviewer}
    reviewersMap = dict(list(zip(reviewers,range(len(reviewers)))))

    #index sequence of large counts, like [[2],[3],..]
    validSeq = np.argwhere(counts>=density)

    #filtered raw list(ID string, ID string,rating number)
    validList = [review for review in rawList if [reviewersMap[review[0]]] in validSeq]
    return validList

def separate(validList):
	#Take the first column of validList (reviewer ID strings)
	reviewersbyreviews = np.array(validList)[:,0]
	#reviewer list without repeat(string)
	reviewers = np.unique(reviewersbyreviews)
	#{reviewerID(string): i}
	reviewersMap = dict(list(zip(reviewers,range(len(reviewers)))))

	#Take the second column of validList (item ID strings)
	itemsbyreviews = np.array(validList)[:,1]
	#item list without repeat(string)
	items = np.unique(itemsbyreviews)
	#{itemID(string): j}
	itemsMap = dict(list(zip(items,range(len(items)))))

	#Update the validList to (i, j, rating number) list, according to reviewersMap and itemsMap
	for i in range(len(validList)):
	    validList[i] = tuple([reviewersMap[validList[i][0]], itemsMap[validList[i][1]], float(validList[i][2])])
	validSet = set(validList)
	testSet = set()

	for reviewer in reviewers:
		#Retrive all the reviews by the current reviewer
		availableRecords = set(review for review in validList if (review[0] == reviewersMap[reviewer]))
		#Randomly take at least 1/5 of the reviews and put them into testSet
		random_sample_test = set(random.sample(availableRecords,density//5+1))
		for i in random_sample_test:
			testSet.add(i)
	#Remaining records go to trainSet
	trainSet = validSet.difference(testSet)
	return validSet, trainSet,testSet, reviewersMap, itemsMap

def dataProcessing():
	validList = findValid()
	validSet, trainSet, testSet, user, item = separate(validList)
	#reviewersMap -> user; itemsMap -> item
	#validSet, trainSet, testSet are sets of (i, j, rating no) tuples
	#Note that validList with ID strings has been replaced

	trainList = np.array(list(trainSet))
	training_len = len(trainList)
	train_user_id = trainList[:,0]# I-pos no of users in training data with duplicated entries
	train_item_id = trainList[:,1]# J-pos no of items in training data with duplicated entries
	train_rating = trainList[:,2]# corresponding ratings
	return user,item,train_user_id,train_item_id,train_rating,testSet

def create_rating_list():
	#### creating dictionary {user i: array[item j, rating r]}.
	ratings_by_i=dict()
	#### creating dictionary {item j: array[user i, rating r]}.
	ratings_by_j=dict()
	
	for key,value in user.iteritems():#key:IDstring; value:i
		pair = np.argwhere(train_user_id==value)#[[2],[5],..]
		ratings_by_i[value]=np.array([(train_item_id[pair[j][0]],train_rating[pair[j][0]]) for j in range(pair.shape[0])]).astype(int)
	for key,value in item.iteritems():
		pair = np.argwhere(train_item_id==value)
		ratings_by_j[value]=np.array([(train_user_id[pair[k][0]],train_rating[pair[k]][0]) for k in range(pair.shape[0])]).astype(int)
	return ratings_by_i,ratings_by_j

#GLOBAL VARIABLES
#density
#user,item,train_user_id,train_item_id,train_rating
#ratings_by_i,ratings_by_j
#testSet

def matrix_fac(K, regCo, ratings_by_i, ratings_by_j):
	#R matrix dimensions
	M = len(ratings_by_i)
	N = len(ratings_by_j)

	mu=sum(train_rating)/len(train_rating)#average rating
	reg=regCo/statistics.pvariance(train_rating,mu)#regularization coefficient

	U = abs(np.random.randn(M, K) / K) #vertical
	V = abs(np.random.randn(K, N) / K) #horizontal
	#r_hat = np.zeros([M,N])
	
	#fill in R using ratings_by_i
	R=np.zeros([M,N])
	for i in ratings_by_i:
		#i=int(i)
		if ratings_by_i[i].shape[0]>0:
			ind_movie=ratings_by_i[i][:,0]#\Omega_i
			ind_rating=ratings_by_i[i][:,1]#corresponding ratings
			#ind_movie=[int(ind_movie[j]) for j in range(ind_movie.shape[0])]
			R[i,ind_movie]=ind_rating.ravel()

	#Iteratively training
	for t in range(100):
		for i in range(M):
			if len(ratings_by_i[i])>0:
				#rate_ind=np.zeros(len(ratings_by_i[i]))
				rate_ind=ratings_by_i[i][:,1]#R_{i,j} where j \in \Omega_i
				#movie_ind=np.zeros(len(ratings_by_i[i]))
				movie_ind=ratings_by_i[i][:,0]#\Omega_i
				#movie_ind=[int(movie_ind[j]) for j in range(movie_ind.shape[0])]
				U[i,:]=(np.linalg.inv(V[:,movie_ind].dot(V[:,movie_ind].T)+reg*np.eye(K)).dot(V[:,movie_ind].dot(rate_ind))).ravel()   
		for j in range(N):
			if len(ratings_by_j[j])>0:
				#rate_ind=np.zeros(len(ratings_by_j[j]))
				rate_ind=ratings_by_j[j][:,1]#R_{i,j} where i \in \Omega_j
				#user_ind=np.zeros(len(ratings_by_j[j]))
				user_ind=ratings_by_j[j][:,0]#\Omega_i
				#user_ind=[int(user_ind[i]) for i in range(user_ind.shape[0])]
				V[:,j]=(np.linalg.inv(U[user_ind,:].T.dot(U[user_ind,:])+reg*np.eye(K)).dot((U[user_ind,:].T.dot(rate_ind)))).ravel()

	r_hat=U.dot(V)
	#Training error
	error = np.mean(abs(R - r_hat))
	print "Training error",error
	return r_hat

def test():
	def facAndTest(K, regCo):
		accuracy = 0
		#matrix factorization and training
		r_hat = matrix_fac(K, regCo,ratings_by_i,ratings_by_j)
		#test error
		for i in range(len(test_data)):
			test_user_id = test_data[i][0]
			test_item_id = test_data[i][1]
			test_result[i] = test_data[i][2]
			test_pred[i] = r_hat[test_user_id,test_item_id]
		error = np.mean(abs(test_result - test_pred))
		print "Test error", error

	test_data = list(testSet)
	test_result = np.zeros(len(test_data))
	test_pred = np.zeros(len(test_data))
	for regCo in [3]:
		for K in [15]:
			print "regularization = ",regCo
			print "K feature",K
			facAndTest(K, regCo)

density = 10
user,item,train_user_id,train_item_id,train_rating,testSet = dataProcessing()
ratings_by_i,ratings_by_j = create_rating_list()
test()	