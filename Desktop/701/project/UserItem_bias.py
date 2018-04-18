import os
import numpy as np
import random
import statistics
from preprocess import preprocess

class collaborative_filter_bias(object):
	def __init__(self,regCo,K,filepath,density):
		self.regCo = regCo
		self.prep_data = preprocess(filepath,density)

		self.user,self.item,self.train_user_id,self.train_item_id,self.train_rating,self.testSet = self.prep_data.dataProcessing()
		self.ratings_by_i,self.ratings_by_j = self.prep_data.create_rating_list()


	def matrix_bias(self,K):
		M = len(self.ratings_by_i)
		N = len(self.ratings_by_j)

		mu=sum(self.train_rating)/len(self.train_rating)#average rating
		reg=self.regCo/statistics.pvariance(self.train_rating,mu)#regularization coefficient

		U = abs(np.random.randn(M, K) / K) #vertical
		V = abs(np.random.randn(K, N) / K) #horizontal
		b_user = np.zeros(M)
		b_item = np.zeros(N)

		R=np.zeros([M,N])
		for i in self.ratings_by_i:
			#i=int(i)
			if self.ratings_by_i[i].shape[0]>0:
				ind_movie=self.ratings_by_i[i][:,0]#\Omega_i
				ind_rating=self.ratings_by_i[i][:,1]#corresponding ratings
				#ind_movie=[int(ind_movie[j]) for j in range(ind_movie.shape[0])]
				R[i,ind_movie]=ind_rating.ravel()

		for t in range(100):

			# update user bias
			for i in range(M):
				if i in self.ratings_by_i:
					accum = 0
					for j, r in self.ratings_by_i[i]:
						accum += (r - U[i,:].dot(V[:,j]) - b_item[j] - mu)
					if len(self.ratings_by_i[i])>0:
						b_user[i] = accum / (1 + reg) / len(self.ratings_by_i[i])

			# update U
			for i in range(M):
				if i in self.ratings_by_i:
					matrix = np.zeros((K, K)) + reg*np.eye(K)
					vector = np.zeros(K)
					for j, r in self.ratings_by_i[i]:
						matrix += np.outer(V[:,j], V[:,j])
						vector += (r - b_user[i] - b_item[j] - mu)*V[:,j]
					U[i,:] = np.linalg.solve(matrix, vector)

			# update item bias
			for j in range(N):
				if j in self.ratings_by_j:
					accum = 0
					for i, r in self.ratings_by_j[j]:
						accum += (r - U[i,:].dot(V[:,j]) - b_user[i] - mu)
					if len(self.ratings_by_j[j])>0:
						b_item[j] = accum / (1 + reg) / len(self.ratings_by_j[j])

			# update V
			for j in range(N):
				if j in self.ratings_by_j:
					matrix = np.zeros((K, K)) + reg*np.eye(K)
					vector = np.zeros(K)
					for i, r in self.ratings_by_j[j]:
						matrix += np.outer(U[i,:], U[i,:])
						vector += (r - b_user[i] - b_item[j] - mu)*U[i,:]
						V[:,j] = np.linalg.solve(matrix, vector)


		r_hat = np.zeros([M,N])
		error = np.zeros([M,N])
		for i in range(M):
			for j in range(N):
				#if R[i,j]>0:
				r_hat[i,j] = U[i,:].dot(V[:,j])+b_user[i]+b_item[j]+mu
				if R[i,j]>0:
					error[i,j]=abs(R[i,j]-r_hat[i,j])

				#print U[i,:].dot(V[:,j])
		error = np.mean(error)
		print "Training error with bias",error
		return r_hat
				



	def test(self,K,regCo):
		def facAndTest(K, regCo):
			accuracy = 0
			#matrix factorization and training
			#r_hat = self.matrix_fac(K)
			r_hat = self.matrix_bias(K)
			#test error
			for i in range(len(test_data)):
				test_user_id = test_data[i][0]
				test_item_id = test_data[i][1]
				test_result[i] = test_data[i][2]
				test_pred[i] = r_hat[test_user_id,test_item_id]
			error = np.mean(abs(test_result - test_pred))
			print "Test error with bias", error
		test_data = list(self.testSet)
		test_result = np.zeros(len(test_data))
		test_pred = np.zeros(len(test_data))
		for Lambda in regCo:
			for k in K:
				print "regularization = ",Lambda
				print "K feature",k
				error = facAndTest(k, Lambda)
		return error