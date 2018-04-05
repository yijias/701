import os
import numpy as np
import random
import statistics

def dataToArray():
	path = os.getcwd()
	file = path + '/ratings_Musical_Instruments.csv'
	f = open(file,'r')
	rawString = f.read()
	rawList = rawString.split('\n')[:-1]
	#print type(rawList)
	user = dict()
	item = dict()
	user_id = 0
	item_id = 0
	random_row = random.sample(xrange(len(rawList)),375000)
	train = set(rawList[x] for x in random_row)
	test_data = list(set(rawList) - train)
	#print type(test)
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
	for row in range(len(rawList)):
		rawList[row] = rawList[row].split(',')
		rawList[row] = tuple(rawList[row][:-1])

		if rawList[row][0] not in user:
			user_id+=1
			user[rawList[row][0]] = user_id
		if rawList[row][1] not in item:
			item_id+=1
			item[rawList[row][1]] = item_id

	for i in range(training_len):
		row = random_row[i]
		train_user_id[i] = user[rawList[row][0]]
		train_item_id[i] = item[rawList[row][1]]
		train_rating[i] = rawList[row][2]


	return user,item,train_user_id,train_item_id,train_rating,test_data

user,item,train_user_id,train_item_id,train_rating,test_data = dataToArray()

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
	
	return r_hat
def test():
	#test_user_id = np.zeros(len(test))
	#test_item_id = np.zeros(len(test))
	test_result = np.zeros(len(test_data))
	test_pred = np.zeros(len(test_data))
	r_hat = matrix_fac()
	for i in range(len(test_data)):
		test_data[i] = test_data[i].split(',')
		test_data[i] = tuple(test_data[i][:-1])
		test_user_id = int(user[test_data[i][0]])
		test_item_id = int(item[test_data[i][1]])
		test_result[i] = test_data[i][2]
		#print test_result[i]
		test_pred[i] = r_hat[test_user_id-1,test_item_id-1]
	sq_error = np.mean((test_result - test_pred)**2)
	print sq_error
test()	