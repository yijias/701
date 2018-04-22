import os
import numpy as np
import random
import statistics
from preprocess import preprocess
from collaborative_filter import collaborative_filter

class collaborative_filter_bias(collaborative_filter):
    def __init__(self,user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j):
        collaborative_filter.__init__(self,user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
    def matrix_fac(self, K, regCo, label):
        M,N,mu,R = self.dataTrans()
        reg=regCo/statistics.pvariance(self.train_rating,mu)
        U = (np.random.randn(M, K)/K) #vertical
        V = (np.random.randn(K, N)/K) #horizontal
        b_user = np.zeros(M)
        b_item = np.zeros(N)
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
        #training error
        error = np.zeros([M,N])
        for i in range(M):
            for j in range(N):
                r_hat[i,j] = U[i,:].dot(V[:,j])+b_user[i]+b_item[j]+mu
                if R[i,j]>0:
                    error[i,j]=abs(R[i,j]-r_hat[i,j])

        error = np.mean(error)
        print("Training error %s" %label,error)
        return r_hat



        