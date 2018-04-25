from preprocess import preprocess
from collaborative_filter import collaborative_filter
from UserItem_bias import collaborative_filter_bias
import os
import numpy as np
import random
import statistics
from integrate_CF import integrated_model_CF

def main():
	density = 50
	path = os.getcwd()
	file = path + '/ratings_Musical_Instruments.csv'
	regCo = [1]
	K = [5]
	#predict = collaborative_filter(regCo,K,file,density)
	#predict_base = collaborative_filter(regCo,K,file,density)
	#r_hat_base = predict_base.test(K,regCo)
	predict_withBias = collaborative_filter_bias(regCo,K,file,density)
	r_hat_withBias = predict_withBias.test(K,regCo)

	#integrate_model = integrated_model_CF(regCo,K,file,density)
	#r_hat_imp = integrate_model.test(K,regCo)
	#test()	

if __name__ == '__main__':
	main()