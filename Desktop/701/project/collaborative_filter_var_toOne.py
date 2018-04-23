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
                r[i,j] = 1/(1+np.exp(-r[i,j]))
        return r
    def sigmas(self, U,V,R):
        M,N = R.shape
        r_hat = self.logistic(U.dot(V),M,N) 
        sigma = self.calSigma(R,r_hat,c=0)
        sigma_U = np.var(np.ravel(U)) #user variance
        sigma_V = np.var(np.ravel(V)) #item variance
        '''
        u,v = np.ravel(U),np.ravel(V)
        mu_u,mu_v = np.sum(u)/len(u),np.sum(v)/len(v)
        u,v = u-mu_u,v-mu_v
        sigma_U = np.mean(u.dot(u))
        sigma_V = np.mean(v.dot(v))
        '''
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

    def test(self,K,regCos,label, iters=2, step = 3):
        def facAndTest(k, regCo):
            #matrix factorization and training
            M,N,mu,R_ori = self.dataTrans()
            R = self.mapToOne(R_ori,M,N) #map R_ori to R -- [1,5] to [0,1]; R will be used throughout training
            U,V = self.initiate(M,N,k)
            sigma, sigma_U, sigma_V = self.sigmas(U,V,R)
            for i in range(1,iters):
                U,V,r_hat,sigma,sigma_U,sigma_V= self.matrix_fac(k,step,M,N,U,V,R,sigma,sigma_U,sigma_V)
                train_error = self.trainError(R_ori,r_hat)
                print("%s step training error %s" %(i*step,label),train_error)
                test_error = self.testError(r_hat, IDpairs, trueValues)
                print("%s step testing error %s" %(i*step,label),test_error)

        test_data = np.array(list(self.testSet))
        trueValues = test_data[:,-1].ravel()
        IDpairs = (test_data[:,:-1].T).astype(int)
        for regCo in regCos:
            for k in K:
                print "regularization = ",regCo
                print "K feature",k
                facAndTest(k, regCo)

