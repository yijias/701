import time
from preprocess import preprocess
from collaborative_filter import collaborative_filter
from collaborative_filter_bias import collaborative_filter_bias
#from collaborative_filter_var import collaborative_filter_var
from collaborative_filter_var_toOne import collaborative_filter_var_toOne
#from UserItem_bias import collaborative_filter_bias
import os
import numpy as np
import random
import statistics

def main():
	density = 20
	path = os.getcwd()
	file = path + '/ratings_Musical_Instruments.csv'
	regCo = [1]
	K = [20]
	data = preprocess(file,density)
	user,item,train_user_id,train_item_id,train_rating,testSet = data.dataProcessing()
	ratings_by_i, ratings_by_j = data.create_rating_list()
	time0 = time.time()
	#predict_base = collaborative_filter(user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
	#r_hat_base = predict_base.test(K,regCo,'base line')
	time1 = time.time()
	predict_withBias = collaborative_filter_bias(user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
	r_hat_withBias = predict_withBias.test(K,regCo,'with bias')
	time2 = time.time()
	#predict_var_toOne = collaborative_filter_var_toOne(user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
	#r_hat_var_toOne = predict_var_toOne.test(K,regCo,'var_toOne')
	time3 = time.time()
	print(time1-time0,time3-time2)
if __name__ == '__main__':
	main()