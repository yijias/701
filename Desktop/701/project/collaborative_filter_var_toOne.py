import os
import numpy as np
import random
import statistics
import copy
import time
from preprocess import preprocess
from collaborative_filter import collaborative_filter
class collaborative_filter_var_toOne(collaborative_filter):
    def __init__(self,user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j):
        collaborative_filter.__init__(self,user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
    
    def calSigma(self,R,r_hat,c=1):
        M = R.shape[0]
        zipped = np.dstack((R,r_hat))
        diff = [[np.square(r1-r2) if r1>=c else 0 for (r1,r2) in zipped[i]] for i in range(M)]
        sigma = np.mean(np.ravel(diff))
        return sigma

    def mapToOne(self,R_ori,M,N):
        R = copy.deepcopy(R_ori)
        for i in range(M):
            for j in range(N):
                R[i,j] = (R_ori[i,j]-1)/(4)
        return R
    def logistic(self,r,M,N):
        for i in range(M):
            for j in range(N):
                r[i,j] = (np.arctan(0.3*r[i,j])+1)/2
                #r[i,j] = 1/(1+np.exp(-0.3*r[i,j]))
        return r
    def sigmas(self, U,V,R):
        M,N = R.shape
        r_hat = self.logistic(U.dot(V),M,N) 
        sigma = self.calSigma(R,r_hat,c=0)
        sigma_U = np.var(np.ravel(U)) #user variance
        sigma_V = np.var(np.ravel(V)) #item variance
        return sigma, sigma_U, sigma_V    

    def matrix_fac(self, K,step,M,N, U,V,R,sigma,sigma_U,sigma_V):
        #Iteratively training
        for t in range(step):
            for i in range(M):
                if len(self.ratings_by_i[i])>0:
                    rate_ind=self.ratings_by_i[i][:,1]#R_{i,j} where j \in \Omega_i
                    movie_ind=self.ratings_by_i[i][:,0]#\Omega_i
                    U[i,:]=(np.linalg.inv(V[:,movie_ind].dot(V[:,movie_ind].T)+(sigma/sigma_U)*np.eye(K)).dot(V[:,movie_ind].dot(rate_ind))).ravel()   
            for j in range(N):
                if len(self.ratings_by_j[j])>0:
                    rate_ind=self.ratings_by_j[j][:,1]#R_{i,j} where i \in \Omega_j
                    user_ind=self.ratings_by_j[j][:,0]#\Omega_i
                    V[:,j]=(np.linalg.inv(U[user_ind,:].T.dot(U[user_ind,:])+(sigma/sigma_V)*np.eye(K)).dot(U[user_ind,:].T.dot(rate_ind))).ravel()
            sigma, sigma_U, sigma_V = self.sigmas(U,V,R)
            #print(sigma, sigma_U, sigma_V)

        r_hat = self.logistic(U.dot(V),M,N)*4+1 #map it back to [1,5]

        return U,V,r_hat,sigma,sigma_U,sigma_V

    def facAndTest(self, k, regCo, iters, step, trueValues, IDpairs,label):
        #matrix factorization and training
        M,N,mu,R_ori = self.dataTrans()
        R = self.mapToOne(R_ori,M,N) #map R_ori to R -- [1,5] to [0,1]; R will be used throughout training
        U,V = self.initiate(M,N,k)
        sigma, sigma_U, sigma_V = self.sigmas(U,V,R)
        
        train = list(); test = list() 
        t0=time.time()
        for i in range(1,iters):
            U,V,r_hat,sigma,sigma_U,sigma_V= self.matrix_fac(k,step,M,N,U,V,R,sigma,sigma_U,sigma_V)
            train_error = self.trainError(R_ori,r_hat)
            print("%s step training error %s" %(i*step,label),train_error)
            train.append([time.time()-t0, train_error])
            test_error = self.testError(r_hat,IDpairs, trueValues)
            print("%s step testing error %s" %(i*step,label),test_error)
            test.append([time.time()-t0, test_error])
        return train, test

