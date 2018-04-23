import os
import numpy as np
import random
import statistics
from preprocess import preprocess

class collaborative_filter(object):
	def __init__(self,user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j):
		self.user,self.item,self.train_user_id,self.train_item_id,self.train_rating,self.testSet = user,item,train_user_id,train_item_id,train_rating,testSet
		self.ratings_by_i,self.ratings_by_j = ratings_by_i,ratings_by_j

	def dataTrans(self):
		#R matrix dimensions
		M = len(self.ratings_by_i)
		N = len(self.ratings_by_j)
		mu=sum(self.train_rating)/len(self.train_rating)#average rating
		
		#fill in R using ratings_by_i
		R=np.zeros([M,N])
		for i in self.ratings_by_i:
			if self.ratings_by_i[i].shape[0]>0:
				ind_movie=self.ratings_by_i[i][:,0]#\Omega_i
				ind_rating=self.ratings_by_i[i][:,1]#corresponding ratings
				R[i,ind_movie]=ind_rating.ravel()
		return M,N,mu,R

	def initiate(self, M,N,K):
		U = (np.random.randn(M, K)/K) #vertical
		V = (np.random.randn(K, N)/K) #horizontal
		return U,V

	def calReg(self,regCo,mu):
		reg=regCo/statistics.pvariance(self.train_rating,mu)
		return reg

	def matrix_fac(self,U,V,M,N, K, reg,step):
		#Iteratively training
		for t in range(step):
			for i in range(M):
				if len(self.ratings_by_i[i])>0:
					rate_ind=self.ratings_by_i[i][:,1]#R_{i,j} where j \in \Omega_i
					movie_ind=self.ratings_by_i[i][:,0]#\Omega_i
					U[i,:]=(np.linalg.inv(V[:,movie_ind].dot(V[:,movie_ind].T)+reg*np.eye(K)).dot(V[:,movie_ind].dot(rate_ind))).ravel()   
			for j in range(N):
				if len(self.ratings_by_j[j])>0:
					rate_ind=self.ratings_by_j[j][:,1]#R_{i,j} where i \in \Omega_j
					user_ind=self.ratings_by_j[j][:,0]#\Omega_i
					V[:,j]=(np.linalg.inv(U[user_ind,:].T.dot(U[user_ind,:])+reg*np.eye(K)).dot((U[user_ind,:].T.dot(rate_ind)))).ravel()
		r_hat = U.dot(V)
		return U,V,r_hat
				
	def test(self,K,regCos,label, iters=20, step = 3):
		def facAndTest(k, regCo):
			#matrix factorization and training
			M,N,mu,R = self.dataTrans()
			U,V = self.initiate(M,N,k)
			reg = self.calReg(regCo,mu)
			for i in range(1,iters):
				U_new,V_new,r_hat = self.matrix_fac(U,V,M,N,k,reg,step)
				U,V = U_new, V_new
				zipped = np.dstack((R,r_hat))
				train_error = [[abs(r1-r2) if r1>=1 else 0 for (r1,r2) in zipped[j]] for j in range(M)]
				train_error = np.mean(train_error)
				print("%s step training error %s" %(i*step,label),train_error)
				test_pred = r_hat[IDpairs[0],IDpairs[1]].ravel()
				test_error = np.mean(abs(test_result - test_pred))
				print("%s step testing error %s" %(i*step,label),test_error)

		test_data = np.array(list(self.testSet))
		test_result = test_data[:,-1].ravel()
		IDpairs = (test_data[:,:-1].T).astype(int)
		print(IDpairs)
		for regCo in regCos:
			for k in K:
				print "regularization = ",regCo
				print "K feature",k
				facAndTest(k, regCo)


