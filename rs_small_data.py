import os
import numpy as np
import random
import statistics
def stringToNumberUser(reviewList):
    reviewers = list(set([review[0] for review in reviewList]))
    #map reviewerID to order number
    reviewersMap = dict(list(zip(reviewers,range(len(reviewers)))))
    return reviewers,reviewersMap

def stringToNumberItem(reviewList):
    items = list(set([review[1] for review in reviewList]))
    #map reviewerID to order number
    itemsMap = dict(list(zip(items,range(len(items)))))
    return items, itemsMap

def findValid():
    path = os.getcwd()
    file = path + '/ratings_Musical_Instruments.csv'
    f = open(file,'r')
    rawString = f.read()
    rawList = rawString.split('\n')[:-1]
    for i in range(len(rawList)):
        rawList[i] = rawList[i].split(',')
        rawList[i] = tuple(rawList[i][:-1])
    #Above got the raw list of all reviews
    #reviewers, reviewersMap= stringToNumberUser(rawList)
    #print(len(reviewers))
    reviewerbyreviews = np.array(rawList)[:,0]
    #count the number of reviews by each person
    reviewers, counts = np.unique(reviewerbyreviews, return_counts=True)  
    reviewersMap = dict(list(zip(reviewers,range(len(reviewers)))))
    validSeq = np.argwhere(counts>=density)
    validList = [review for review in rawList if [reviewersMap[review[0]]] in validSeq]
    return validList

def separate(validList):
	reviewersbyreviews = np.array(validList)[:,0]#string
	reviewers = np.unique(reviewersbyreviews)#string
	reviewersMap = dict(list(zip(reviewers,range(len(reviewers)))))
	itemsbyreviews = np.array(validList)[:,1]
	items = np.unique(itemsbyreviews)
	itemsMap = dict(list(zip(items,range(len(items)))))
	for i in range(len(validList)):
	    validList[i] = tuple([reviewersMap[validList[i][0]], itemsMap[validList[i][1]], float(validList[i][2])])
	validSet = set(validList)
	testSet = set()
	for reviewer in reviewers:
<<<<<<< HEAD
		availableRecords = [review for review in validList if (review[0] == reviewersMap[reviewer])]
		testSet.add(random.sample(availableRecords,density/5))
=======
		availableRecords = set(review for review in validList if (review[0] == reviewersMap[reviewer]))
		random_sample_test = set(random.sample(availableRecords,density/5))
		for i in random_sample_test:
			#print type(i)
			testSet.add(i)
		#testSet.add(set(random.sample(availableRecords,density/5)))
>>>>>>> master
	trainSet = validSet.difference(testSet)
	return validSet, trainSet,testSet, reviewersMap, itemsMap

def dataProcessing():
	validList = findValid()#valid records with strings as ID's
	validSet, trainSet, testSet, user, item = separate(validList)
	# train__id is the id from training data, they are duplicated entries, 
	# one user can rate multiple items
	# user maps user string to number
	# item_id contains unique item's id
	trainList = np.array(list(trainSet))
	training_len = len(trainList)
	train_user_id = trainList[:,0]
	train_item_id = trainList[:,1]
	train_rating = trainList[:,2]
	return user,item,train_user_id,train_item_id,train_rating,testSet

def create_rating_list():
	#### creating dictionary for the set of all user i that have rated item j with rating r.
	ratings_by_i=dict()
	#### creating dictionary for the set of all item j that have usr i have rated with rating r
	ratings_by_j=dict()
	
	for key,value in user.iteritems():
		pair=np.argwhere(train_user_id==value)
		ratings_by_i[value]=np.array([(train_item_id[pair[j][0]],train_rating[pair[j][0]]) for j in range(pair.shape[0])]).astype(int)
	for key,value in item.iteritems():
		pair = np.argwhere(train_item_id==value)
		ratings_by_j[value]=np.array([(train_user_id[pair[k][0]],train_rating[pair[k]][0]) for k in range(pair.shape[0])]).astype(int)
	#print(ratings_by_i)
	#print(ratings_by_j)
	return ratings_by_i,ratings_by_j

def matrix_fac(K, regCo, ratings_by_i, ratings_by_j):
	M = len(ratings_by_i)
	N = len(ratings_by_j)
	#K = 15
	mu=sum(train_rating)/len(train_rating)
	reg=regCo/statistics.pvariance(train_rating,mu)
	#print(reg)
	U = np.random.randn(M, K) / K
	V = np.random.randn(K, N) / K
	r_hat=np.zeros([M,N])
	
	Q=np.zeros([M,N])
	for i in ratings_by_i:
		i=int(i)
		if ratings_by_i[i].shape[0]>0:
			ind_movie=ratings_by_i[i][:,0]
			ind_rating=ratings_by_i[i][:,1]
			ind_movie=[int(ind_movie[j]) for j in range(ind_movie.shape[0])]
			#print(ind_movie)
			Q[i,ind_movie]=ind_rating.ravel()

	#print(R_pre.shape)

	
	for t in range(100):
		for i in range(M):
			if len(ratings_by_i[i])>0:
				rate_ind=np.zeros(len(ratings_by_i[i]))
				#print(ratings_by_i[i+1].shape)
				rate_ind=ratings_by_i[i][:,1]
				movie_ind=np.zeros(len(ratings_by_i[i]))
				movie_ind=ratings_by_i[i][:,0]
				movie_ind=[int(movie_ind[j]) for j in range(movie_ind.shape[0])]
				U[i,:]=np.linalg.inv(V[:,movie_ind].dot(V[:,movie_ind].T)+reg*np.eye(K)).dot(V[:,movie_ind].dot(rate_ind)).ravel()       
		for j in range(N):
			if len(ratings_by_j[j])>0:
				rate_ind=np.zeros(len(ratings_by_j[j]))
				rate_ind=ratings_by_j[j][:,1]
				user_ind=np.zeros(len(ratings_by_j[j]))
				user_ind=ratings_by_j[j][:,0]
				user_ind=[int(user_ind[i]) for i in range(user_ind.shape[0])]
				V[:,j]=np.linalg.inv(U[user_ind,:].T.dot(U[user_ind,:])+reg*np.eye(K)).dot((U[user_ind,:].T.dot(rate_ind))).ravel()
	r_hat=U.dot(V)
	error = np.mean(abs(Q - r_hat))
	print "Training error",error
	return r_hat
def test():
	#test_user_id = np.zeros(len(test))
	#test_item_id = np.zeros(len(test))
	test_data = list(testSet)
	#print test_data[0]

	test_result = np.zeros(len(test_data))
	test_pred = np.zeros(len(test_data))
	
	def facAndTest(K, regCo):
		accuracy = 0
		r_hat = matrix_fac(K, regCo,ratings_by_i,ratings_by_j)
		
		for i in range(len(test_data)):
			#print(test_data[i][0])
			test_user_id = test_data[i][0]
			test_item_id = test_data[i][1]
			test_result[i] = test_data[i][2]

			test_pred[i] = r_hat[test_user_id,test_item_id]
<<<<<<< HEAD
			#true = test_result[i]
			#if (true == int(test_pred[i]) or true == int(test_pred[i])+1 or true == int(test_pred[i])-1):accuracy+=1
		print(test_result)
		print(test_pred)
		error = np.sum(abs(test_result - test_pred))/len(test_result)
		#error = float(accuracy)/len(test_result)
		#print sq_error
		print error'''
		train_pred = [0]*len(train_user_id)
		for i in range(len(train_user_id)):
			#print(train_user_id[i],train_item_id[i])
			train_pred[i] = r_hat[int(train_user_id[i]),int(train_item_id[i])]
		error = np.sum(abs(train_rating - train_pred))/len(train_rating)
		print(error)
	for regCo in range(110,200,10):
		facAndTest(15, regCo)
density = 50
=======

		error = np.mean(abs(test_result - test_pred))
		print "Test error",error

	for regCo in [0.1,0.5,1,1.5,2]:
		print "regularization = ",regCo
		for K in [5,10,15]:
			print "K feature",K
			facAndTest(K, regCo)

density = 10
>>>>>>> master
user,item,train_user_id,train_item_id,train_rating,testSet = dataProcessing()
ratings_by_i,ratings_by_j = create_rating_list()
test()	