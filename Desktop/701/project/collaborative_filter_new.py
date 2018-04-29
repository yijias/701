import os
import numpy as np
import random
import statistics
import time
from preprocess import preprocess
from collaborative_filter_bias import collaborative_filter_bias

class collaborative_filter_new(collaborative_filter_bias):
    def __init__(self,user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j, lr = 0.007):
        collaborative_filter_bias.__init__(self,user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
        self.lr = lr
        trainList = list(np.stack((train_user_id,train_item_id,train_rating), axis = -1))
        trainTuples = [tuple(record) for record in trainList]
        self.trainSet = set(trainTuples)

    def initiate(self, M,N,K):
        U = (np.random.randn(M, K)/K) #vertical
        V = (np.random.randn(K, N)/K) #horizontal
        b_user = np.zeros(M)
        b_item = np.zeros(N)       
        y = np.zeros([K,N]) + 0.1 # implicit rating
        return U,V,b_user,b_item,y

    def matrix_fac(self,U,V,R,M,N,y,b_user,b_item,mu, K,reg,step):
    	for t in range(step):
    		for u, i, r in self.trainSet:
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

    	r_hat = (U+y_u).dot(V) + np.tile(b_user,(N,1)).T + np.tile(b_item,(M,1)) + mu
    	return U,V,b_user,b_item,y,r_hat

    def facAndTest(self, k, regCo, iters, step, trueValues, IDpairs,label):
        #matrix factorization and training
        M,N,mu,R = self.dataTrans()
        U,V,b_user,b_item,y = self.initiate(M,N,k)
        reg = self.calReg(regCo,mu)

        train = list(); test = list()
        t0=time.time()
        for i in range(1,iters):
            U,V,b_user,b_item,y,r_hat = self.matrix_fac(U,V,R,M,N,y,b_user,b_item,mu, k,reg,step)
            train_error = self.trainError(R,r_hat)
            print("%s step training error %s" %(i*step,label),train_error)
            train.append([time.time()-t0, train_error])
            test_error = self.testError(r_hat,IDpairs, trueValues)
            print("%s step testing error %s" %(i*step,label),test_error)
            test.append([time.time()-t0, test_error])
        return train, test
