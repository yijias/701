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
        self.trainSet = set(list(np.stack((train_user_id,train_item_id,train_rating), axis = -1)))
    def matrix_fac(self,U,V,M,N,b_user,b_item,mu, K,reg,step):
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
    			r_hat = U.dot(V) + np.tile(b_user,(N,1)).T + np.tile(b_item,(M,1)) + mu
    	return U,V,b_user,b_item,r_hat
