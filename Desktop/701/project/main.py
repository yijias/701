from preprocess import preprocess
from collaborative_filter import collaborative_filter
from collaborative_filter_bias import collaborative_filter_bias
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
	predict_base = collaborative_filter(user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
	r_hat_base = predict_base.test(K,regCo)
	predict_withBias = collaborative_filter_bias(user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
	r_hat_withBias = predict_withBias.test(K,regCo)

if __name__ == '__main__':
	main()