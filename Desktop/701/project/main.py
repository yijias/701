import time
import matplotlib.pyplot as plt
from preprocess import preprocess
from collaborative_filter import collaborative_filter
from collaborative_filter_bias import collaborative_filter_bias
from collaborative_filter_var_toOne import collaborative_filter_var_toOne
import os
import numpy as np
import random
import statistics

def plotError(data, title='', output_path=None, file_name=None, legends = None):
    fig = plt.figure()
    num = len(data)
    for i in range(num):
    	plt.plot(data[i].T[0]-data[i].T[0][0],data[i].T[1],label = legends[i])
    plt.legend()
    plt.title(title)
    plt.show()
    if output_path is not None and file_name is not None:
        fig.savefig(os.path.join(output_path, file_name))

def main():
	density = 20
	path = os.getcwd()
	file = path + '/ratings_Musical_Instruments.csv'
	regCo = [1]
	K = [10]
	data = preprocess(file,density)
	user,item,train_user_id,train_item_id,train_rating,testSet = data.dataProcessing()
	ratings_by_i, ratings_by_j = data.create_rating_list()

	time0 = time.time()

	predict_base = collaborative_filter(user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
	train_base, test_base = predict_base.test(K,regCo,'base line',iters = 20, step = 5)
	time1 = time.time()

	predict_withBias = collaborative_filter_bias(user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
	train_withBias, test_withBias = predict_withBias.test(K,regCo,'with bias',iters=10,step=3)
	time2 = time.time()

	predict_var_toOne = collaborative_filter_var_toOne(user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
	train_var_toOne, test_var_toOne = predict_var_toOne.test(K,regCo,'var_toOne',iters=6,step=1)
	time3 = time.time()

	print(time1-time0,time2-time1,time3-time2)
	for regco in regCo:
		for k in K:
			train = np.array([np.array(train_base[regco][k]),np.array(train_withBias[regco][k]),np.array(train_var_toOne[regco][k])])
			test = np.array([np.array(test_base[regco][k]),np.array(test_withBias[regco][k]),np.array(test_var_toOne[regco][k])])
			#xy = np.asarray([train_base[regco][k],train_withBias[regco][k],train_var_toOne[regco][k]])
			legends = ['base','withBias','with variance']
			plotError(train, title = 'Train Errors',output_path=os.getcwd(), file_name = 'trains_%s_%s_%s.jpg' %(density, regco, k), legends = legends)
			plotError(test, title = 'Test Errors',output_path=os.getcwd(), file_name = 'test_%s_%s_%s.jpg' %(density, regco, k), legends = legends)


if __name__ == '__main__':
	main()