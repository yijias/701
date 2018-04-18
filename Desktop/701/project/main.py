from preprocess import preprocess
from collaborative_filter import collaborative_filter
from UserItem_bias import collaborative_filter_bias
import os
import numpy as np
import random
import statistics

def main():
	density = 10
	path = os.getcwd()
	file = path + '/ratings_Musical_Instruments.csv'
	regCo = [1]
	K = [5]
	#predict = collaborative_filter(regCo,K,file,density)
	predict = collaborative_filter_bias(regCo,K,file,density)
	r_hat = predict.test(K,regCo)

	#test()	

if __name__ == '__main__':
	main()