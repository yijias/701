import os
import numpy as np
import random
import statistics
import time
from preprocess import preprocess
from collaborative_filter import collaborative_filter

class collaborative_filter_bias(collaborative_filter):
    def __init__(self,user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j):
        collaborative_filter.__init__(self,user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
    
    def initiate(self, M,N,K):
        U = (np.random.randn(M, K)/K) #vertical
        V = (np.random.randn(K, N)/K) #horizontal
        b_user = np.zeros(M)
        b_item = np.zeros(N)       
        return U,V,b_user,b_item 

    def matrix_fac(self,U,V,M,N,b_user,b_item,mu, K,reg,step):
        for t in range(step):
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
        r_hat = U.dot(V) + np.tile(b_user,(N,1)).T + np.tile(b_item,(M,1)) + mu
        return U,V,b_user,b_item,r_hat

    def facAndTest(self, k, regCo, iters, step, trueValues, IDpairs,label):
        #matrix factorization and training
        M,N,mu,R = self.dataTrans()
        U,V,b_user,b_item = self.initiate(M,N,k)
        reg = self.calReg(regCo,mu)

        train = list(); test = list()
        t0=time.time()
        for i in range(1,iters):
            U,V,b_user,b_item,r_hat = self.matrix_fac(U,V,M,N,b_user,b_item,mu, k,reg,step)
            train_error = self.trainError(R,r_hat)
            print("%s step training error %s" %(i*step,label),train_error)
            train.append([time.time()-t0, train_error])
            test_error = self.testError(r_hat,IDpairs, trueValues)
            print("%s step testing error %s" %(i*step,label),test_error)
            test.append([time.time()-t0, test_error])
        return train, test




        