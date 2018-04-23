import os
import numpy as np
import random
import statistics
import copy
from preprocess import preprocess
from collaborative_filter_var import collaborative_filter_var
class collaborative_filter_var_toOne(collaborative_filter_var):
    def __init__(self,user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j):
        collaborative_filter_var.__init__(self,user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
    
    def mapToOne(self,R_ori,M,N):
        R = copy.deepcopy(R_ori)
        for i in range(M):
            for j in range(N):
                R[i,j] = (R_ori[i,j]-1)/(4)
        return R
    def logistic(self,r,M,N):
        for i in range(M):
            for j in range(N):
                r[i,j] = 1/(1+np.exp(-r[i,j]))
        return r

    def matrix_fac(self, K, regCo, label):
        M,N,mu,R_ori = self.dataTrans()
        R = self.mapToOne(R_ori,M,N) #map R_ori to R -- [1,5] to [0,1]; R will be used throughout training
        U = (np.random.randn(M, K)) #vertical
        V = (np.random.randn(K, N)) #horizontal
        r_hat = self.logistic(U.dot(V),M,N) 

        sigma = self.calSigma(R,r_hat,c=0)
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
            
            r_hat = self.logistic(U.dot(V),M,N)
            sigma = self.calSigma(R,r_hat,c=0)
            sigma_U = statistics.pvariance(np.ravel(U)) #user variance
            sigma_V = statistics.pvariance(np.ravel(V)) #item variance
            print(sigma, sigma_U, sigma_V)

        r_hat = self.logistic(U.dot(V),M,N)*4+1 #map it back to [1,5]

        #training error; using R_ori
        zipped = np.dstack((R_ori,r_hat))
        error = [[abs(r1-r2) if r1>=1 else 0 for (r1,r2) in zipped[i]] for i in range(M)]
        error = np.mean(np.array(error))        
        print("Training error %s" %label,error)

        return r_hat


