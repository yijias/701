import os
import numpy as np
import random

def dataToArray():
	path = os.getcwd()
	file = path + '/ratings_Musical_Instruments.csv'
	f = open(file,'r')
	rawString = f.read()
	rawList = rawString.split('\n')[:-1]
	user = dict()
	item = dict()
	user_id = 0
	item_id = 0
	random_row = random.sample(xrange(len(rawList)),375000)
	
	#print len(random_row)
	# train__id is the id from training data, they are duplicated entries, 
	# one user can rate multiple items
	# user_id contains unique user's id
	# item_id contains unique item's id
	training_len = len(random_row)
	train_user_id=np.zeros(training_len)
	train_item_id=np.zeros(training_len)
	train_rating=np.zeros(training_len)
	#print len(train_user_id)
	#for i in random_row:
	for i in range(training_len):
		row = random_row[i]
		rawList[row] = rawList[row].split(',')
		rawList[row] = tuple(rawList[row][:-1])

		if rawList[row][0] not in user:
			user_id+=1
			user[rawList[row][0]] = user_id
		if rawList[row][1] not in item:
			item_id+=1

			item[rawList[row][1]] = item_id

		train_user_id[i] = user[rawList[row][0]]
		train_item_id[i] = item[rawList[row][1]]
		train_rating[i] = rawList[row][2]


	return user,item,train_user_id,train_item_id,train_rating

user,item,train_user_id,train_item_id,train_rating = dataToArray()

def create_rating_list():
	#### creating dictionary for the set of all user i that have rated item j with rating r.
	ratings_by_i=dict()
	#### creating dictionary for the set of all item j that have usr i have rated with rating r
	ratings_by_j=dict()
	
	for key,value in user.iteritems():
		pair=np.argwhere(train_user_id==value)
		#print pair
		ratings_by_i[value]=np.array([[train_item_id[pair][j],train_rating[pair][j]] for j in range(pair.shape[0])])
	for key,value in item.iteritems():
		pair = np.argwhere(train_item_id==value)
		#print pair
		ratings_by_j[value]=np.asarray([[train_user_id[pair][k],train_rating[pair][k]] for k in range(pair.shape[0])])
	return ratings_by_i,ratings_by_j

def matrix_fac():
	ratings_by_,ratings_by_j = create_rating_list()
	M = len(user)
	N = len(item)
	K = 5
	U = np.random.randn(M, K) / K
	V = np.random.randn(K, N) / K
	B = np.zeros(M)
	C = np.zeros(N)
	r_hat=np.zeros([M,N])
	T = 100 # 100 epochs for now
	for t in xrange(T):
		# update B
		for i in xrange(M):
			if i in ratings_by_i:
				accum = 0
				for j, r in ratings_by_i[i]:
					accum += (r - U[i,:].dot(V[:,j]) - C[j] - mu)
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