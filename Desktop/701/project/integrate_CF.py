import os
import numpy as np
import random
import statistics
from preprocess import preprocess

class integrated_model_CF(object):
	def __init__(self,regCo,K,filepath,density,lr=0.007):
		self.regCo = regCo
		self.prep_data = preprocess(filepath,density)

		self.user,self.item,self.train_user_id,self.train_item_id,self.train_rating,self.testSet,self.trainSet = self.prep_data.dataProcessing()
		self.ratings_by_i,self.ratings_by_j = self.prep_data.create_rating_list()
		self.lr = lr

	def matrix_bias(self,K):
		M = len(self.ratings_by_i) # num of users
		N = len(self.ratings_by_j)
		y = np.zeros([K,N]) + 0.1 # implicit rating
		
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
		print "finish fill in rating matrix"
		for t in range(100):
			for u, i, r in self.trainSet:
				#print u,i,r
				Nu = self.ratings_by_i[int(u)]
				I_Nu = len(Nu)
				u = int(u)
				i = int(i)
				sqrt_Nu = np.sqrt(I_Nu)
				y_u = np.sum(y[:,Nu])

				u_impl_prf = y_u/sqrt_Nu
				rp = mu + b_user[u] + b_item[i]+(U[u,:]+u_impl_prf).dot(V[:,i])
				e_ui = R[u,i] - rp

				b_item[i] += self.lr*(e_ui-reg*b_item[i])
				V[:,i] += self.lr*(e_ui*(U[u,:]+u_impl_prf) - reg*V[:,i])
				b_user[u] += self.lr*(e_ui - reg*b_user[u])
				U[u,:] += self.lr * (e_ui*V[:,i] - reg*U[u,:])
				for p in Nu:
					y[:,p] += self.lr*(e_ui * V[:,p]/sqrt_Nu - reg * y[:,p])
			print "processing epoch {}".format(t),e_ui
		print "finish updating parameters"
		r_hat = np.zeros([M,N])
		error = np.zeros([M,N])
		for i in range(M):
			for j in range(N):
				#if R[i,j]>0:
				Nu = self.ratings_by_i[i]
				I_Nu = len(Nu)
				sqrt_Nu = np.sqrt(I_Nu)
				y_u = np.sum(y[:,Nu])/sqrt_Nu

				r_hat[i,j] = (U[i,:]+y_u).dot(V[:,j])+b_user[i]+b_item[j]+mu
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
			error = np.sqrt(np.mean((abs(test_result - test_pred))**2))
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