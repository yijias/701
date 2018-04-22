import os
import numpy as np
import random
import statistics
from preprocess import preprocess
from collaborative_filter import collaborative_filter

class collaborative_filter_var(collaborative_filter):
    def __init__(self,user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j):
        collaborative_filter.__init__(self,user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
    
    def calSigma(self,R,r_hat,c=1):
        M = R.shape[0]
        zipped = np.dstack((R,r_hat))
        diff = [[np.square(r1-r2) if r1>=c else 0 for (r1,r2) in zipped[i]] for i in range(M)]
        sigma = np.mean(np.ravel(diff))
        return sigma

    def matrix_fac(self, K, regCo, label):
        M,N,mu,R = self.dataTrans()
        U = (np.random.randn(M, K)) #vertical
        V = (np.random.randn(K, N)) #horizontal
        r_hat = U.dot(V)
        sigma = self.calSigma(R,r_hat,c=1)
        sigma_U = statistics.pvariance(np.ravel(U)) #user variance
        sigma_V = statistics.pvariance(np.ravel(V)) #item variance
        print(sigma, sigma_U, sigma_V)
        #Iteratively training
        for t in range(5):
            print(t)
            for i in range(M):
                if len(self.ratings_by_i[i])>0:
                    rate_ind=self.ratings_by_i[i][:,1]#R_{i,j} where j \in \Omega_i
                    movie_ind=self.ratings_by_i[i][:,0]#\Omega_i
                    U[i,:]=(np.linalg.inv(V[:,movie_ind].dot(V[:,movie_ind].T)+regCo*(sigma/sigma_U)*np.eye(K)).dot(V[:,movie_ind].dot(rate_ind))).ravel()   
            for j in range(N):
                if len(self.ratings_by_j[j])>0:
                    rate_ind=self.ratings_by_j[j][:,1]#R_{i,j} where i \in \Omega_j
                    user_ind=self.ratings_by_j[j][:,0]#\Omega_i
                    V[:,j]=(np.linalg.inv(U[user_ind,:].T.dot(U[user_ind,:])+regCo*(sigma/sigma_V)*np.eye(K)).dot(U[user_ind,:].T.dot(rate_ind))).ravel()
            r_hat = U.dot(V)
            sigma = self.calSigma(R,r_hat,c=1)
            sigma_U = statistics.pvariance(np.ravel(U)) #user variance
            sigma_V = statistics.pvariance(np.ravel(V)) #item variance
            print(sigma, sigma_U, sigma_V)

        r_hat = U.dot(V)
        #training error
        zipped = np.dstack((R,r_hat))
        error = [[np.square(r1-r2) if r1>=1 else 0 for (r1,r2) in zipped[i]] for i in range(M)]
        error = np.mean(np.array(error))        
        print("Training error %s" %label,error)
        return r_hat


