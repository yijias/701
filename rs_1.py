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
    validSeq = np.argwhere(counts>50)
    validList = [review for review in rawList if [reviewersMap[review[0]]] in validSeq]
    return validList

def separate(validList):
	reviewersbyreviews = np.array(validList)[:,0]
	reviewers = np.unique(reviewersbyreviews)
	reviewersMap = dict(list(zip(reviewers,range(len(reviewers)))))
	itemsbyreviews = np.array(validList)[:,1]
	items = np.unique(itemsbyreviews)
	itemsMap = dict(list(zip(items,range(len(items)))))
	for i in range(len(validList)):
	    validList[i] = tuple([reviewersMap[validList[i][0]], itemsMap[validList[i][1]], float(validList[i][2])])
	validSet = set(validList)
	testSet = set()
	for reviewer in reviewers:
	    testSet.add(validList[list(reviewersbyreviews).index(reviewer)])
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
	return ratings_by_i,ratings_by_j

def matrix_fac():
	ratings_by_i,ratings_by_j = create_rating_list()
	M = len(user)
	N = len(item)
	K = 5
	mu=sum(train_rating)/len(train_rating)
	reg=1/statistics.pvariance(train_rating,mu)
	U = np.random.randn(M, K) / K
	V = np.random.randn(K, N) / K
	#r_hat=np.zeros([M,N])
	"""
	Q=np.zeros([M,N])
	for i in ratings_by_i:
		i=int(i)
		if ratings_by_i[i].shape[0]>0:
			ind_movie=ratings_by_i[i][:,0]
			ind_rating=ratings_by_i[i][:,1]
			ind_movie=[int(ind_movie[j]-1) for j in range(ind_movie.shape[0])]
			#print(ind_movie)
			Q[i-1,ind_movie]=ind_rating.ravel()

	#print(R_pre.shape)

	
	for t in range(100):
		for i in range(M):
			if ratings_by_i[i+1].shape[0]>0:
				rate_ind=np.zeros(ratings_by_i[i+1].shape[0])
				#print(ratings_by_i[i+1].shape)
				rate_ind=ratings_by_i[i+1][:,1]
				movie_ind=np.zeros(ratings_by_i[i+1].shape[0])
				movie_ind=ratings_by_i[i+1][:,0]
				movie_ind=[int(movie_ind[j]-1) for j in range(movie_ind.shape[0])]
				U[i,:]=np.linalg.inv(V[:,movie_ind].dot(V[:,movie_ind].T)+reg*np.eye(K)).dot(V[:,movie_ind].dot(rate_ind)).ravel()       
		for j in range(N):
			if ratings_by_j[j+1].shape[0]>0:
				rate_ind=np.zeros(ratings_by_j[j+1].shape[0])
				rate_ind=ratings_by_j[j+1][:,1]
				user_ind=np.zeros(ratings_by_j[j+1].shape[0])
				user_ind=ratings_by_j[j+1][:,0]
				user_ind=[int(user_ind[i]-1) for i in range(user_ind.shape[0])]
				V[:,j]=np.linalg.inv(U[user_ind,:].T.dot(U[user_ind,:])+reg*np.eye(K)).dot((U[user_ind,:].T.dot(rate_ind))).ravel()
		r_hat=U.dot(V)
		rmse=np.sqrt(np.mean(pow(r_hat-Q,2)))
	"""	
	B = np.zeros(M)
	C = np.zeros(N)
	r_hat=np.zeros([M,N])
	Q=np.zeros([M,N])
	T = 100 # 100 epochs for now
	for t in xrange(T):
		# update B
		for i in xrange(M):
			if i in ratings_by_i:
				accum = 0
				for j, r in ratings_by_i[i]:
					accum += (r - U[i,:].dot(V[:,int(j)]) - C[int(j)] - mu)
				B[i] = accum / (1 + reg) / len(ratings_by_i[i])

	  # update U
	for i in xrange(M):
		if i in ratings_by_i:
			matrix = np.zeros((K, K)) + reg*np.eye(K)
			vector = np.zeros(K)
			for j, r in ratings_by_i[i]:
				matrix += np.outer(V[:,j], V[:,j])
				vector += (r - B[i] - C[j] - mu)*V[:,j]
			U[i,:] = np.linalg.solve(matrix, vector)

	  # update C
	for j in xrange(N):
		if j in ratings_by_j:
			accum = 0
			for i, r in ratings_by_j[j]:
				accum += (r - U[i,:].dot(V[:,j]) - B[i] - mu)
			print(len(ratings_by_j[j]))
			C[j] = accum / (1 + reg) / len(ratings_by_j[j])

	  # update V
	for j in xrange(N):
		if j in ratings_by_j:
			matrix = np.zeros((K, K)) + reg*np.eye(K)
			vector = np.zeros(K)
			for i, r in ratings_by_j[j]:
				matrix += np.outer(U[i,:], U[i,:])
				vector += (r - B[i] - C[j] - mu)*U[i,:]
			V[:,j] = np.linalg.solve(matrix, vector)
	r_hat=U.dot(V)
	rmse=np.sqrt(np.mean(pow(r_hat-Q,2)))
	
	return r_hat
def test():
	#test_user_id = np.zeros(len(test))
	#test_item_id = np.zeros(len(test))
	test_data = list(testSet)
	test_result = np.zeros(len(test_data))
	test_pred = np.zeros(len(test_data))
	r_hat = matrix_fac()
	for i in range(len(test_data)):
		#print(test_data[i][0])
		test_user_id = test_data[i][0]
		test_item_id = test_data[i][1]
		test_result[i] = test_data[i][2]
		#print test_result[i]
		test_pred[i] = r_hat[test_user_id-1,test_item_id-1]
	sq_error = np.mean((test_result - test_pred)**2)
	print sq_error

user,item,train_user_id,train_item_id,train_rating,testSet = dataProcessing()
test()	