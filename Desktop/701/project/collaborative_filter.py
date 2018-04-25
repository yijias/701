import os
import numpy as np
import random
import statistics
from preprocess import preprocess

class collaborative_filter(object):
	def __init__(self,regCo,K,filepath,density):
		self.regCo = regCo
		self.prep_data = preprocess(filepath,density)

		self.user,self.item,self.train_user_id,self.train_item_id,self.train_rating,self.testSet = self.prep_data.dataProcessing()
		self.ratings_by_i,self.ratings_by_j = self.prep_data.create_rating_list()




	def matrix_fac(self, K):

		#R matrix dimensions

		M = len(self.ratings_by_i)
		N = len(self.ratings_by_j)

		mu=sum(self.train_rating)/len(self.train_rating)#average rating
		reg=self.regCo/statistics.pvariance(self.train_rating,mu)#regularization coefficient

		U = abs(np.random.randn(M, K) / K) #vertical
		V = abs(np.random.randn(K, N) / K) #horizontal
		#r_hat = np.zeros([M,N])
		
		#fill in R using ratings_by_i
		R=np.zeros([M,N])
		for i in self.ratings_by_i:
			#i=int(i)
			if self.ratings_by_i[i].shape[0]>0:
				ind_movie=self.ratings_by_i[i][:,0]#\Omega_i
				ind_rating=self.ratings_by_i[i][:,1]#corresponding ratings
				#ind_movie=[int(ind_movie[j]) for j in range(ind_movie.shape[0])]
				R[i,ind_movie]=ind_rating.ravel()

		#Iteratively training
		for t in range(100):
			for i in range(M):
				if len(self.ratings_by_i[i])>0:
					#rate_ind=np.zeros(len(ratings_by_i[i]))
					rate_ind=self.ratings_by_i[i][:,1]#R_{i,j} where j \in \Omega_i
					#movie_ind=np.zeros(len(ratings_by_i[i]))
					movie_ind=self.ratings_by_i[i][:,0]#\Omega_i
					#movie_ind=[int(movie_ind[j]) for j in range(movie_ind.shape[0])]
					U[i,:]=(np.linalg.inv(V[:,movie_ind].dot(V[:,movie_ind].T)+reg*np.eye(K)).dot(V[:,movie_ind].dot(rate_ind))).ravel()   
			for j in range(N):
				if len(self.ratings_by_j[j])>0:
					#rate_ind=np.zeros(len(ratings_by_j[j]))
					rate_ind=self.ratings_by_j[j][:,1]#R_{i,j} where i \in \Omega_j
					#user_ind=np.zeros(len(ratings_by_j[j]))
					user_ind=self.ratings_by_j[j][:,0]#\Omega_i
					#user_ind=[int(user_ind[i]) for i in range(user_ind.shape[0])]
					V[:,j]=(np.linalg.inv(U[user_ind,:].T.dot(U[user_ind,:])+reg*np.eye(K)).dot((U[user_ind,:].T.dot(rate_ind)))).ravel()

		r_hat = np.zeros([M,N])
		error = np.zeros([M,N])
		for i in range(M):
			for j in range(N):
				#if R[i,j]>0:
				r_hat[i,j] = U[i,:].dot(V[:,j])
				if R[i,j]>0:
					error[i,j]=abs(R[i,j]-r_hat[i,j])

				#print U[i,:].dot(V[:,j])
		error = np.mean(error)

		print "Training error base line",error
		return r_hat
				
	def test(self,K,regCo):
		def facAndTest(K, regCo):
			accuracy = 0
			#matrix factorization and training
			r_hat = self.matrix_fac(K)
			#r_hat = self.matrix_bias(K)
			#test error
			for i in range(len(test_data)):
				test_user_id = test_data[i][0]
				test_item_id = test_data[i][1]
				test_result[i] = test_data[i][2]
				test_pred[i] = r_hat[test_user_id,test_item_id]
			error = np.mean(abs(test_result - test_pred))
			print "Test error base line", error
		test_data = list(self.testSet)
		test_result = np.zeros(len(test_data))
		test_pred = np.zeros(len(test_data))
		for Lambda in regCo:
			for k in K:
				print "regularization = ",Lambda
				print "K feature",k
				error = facAndTest(k, Lambda)
		return error