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

def plotError(data, title='', output_path=os.getcwd(), file_name=None, colors=None, legends = None):
    #fig = plt.figure()
    num = len(data)
    for i in range(num):
    	plt.plot(data[i].T[0],data[i].T[1], color=colors[i])
    	plt.legend(legends[i])
	plt.title(title)
	plt.show()
    #if output_path is not None and file_name is not None:
    #    fig.savefig(os.path.join(output_path, file_name))

def main():
	density = 100
	path = os.getcwd()
	file = path + '/ratings_Musical_Instruments.csv'
	regCo = [1]
	K = [50]
	data = preprocess(file,density)
	user,item,train_user_id,train_item_id,train_rating,testSet = data.dataProcessing()
	ratings_by_i, ratings_by_j = data.create_rating_list()

	time0 = time.time()

	predict_base = collaborative_filter(user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
	train_base, test_base = predict_base.test(K,regCo,'base line',iters = 10, step = 5)
	time1 = time.time()

	predict_withBias = collaborative_filter_bias(user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
	train_withBias, test_withBias = predict_withBias.test(K,regCo,'with bias',iters=3,step=3)
	time2 = time.time()

	predict_var_toOne = collaborative_filter_var_toOne(user,item,train_user_id,train_item_id,train_rating,testSet,ratings_by_i,ratings_by_j)
	train_var_toOne, test_var_toOne = predict_var_toOne.test(K,regCo,'var_toOne',iters=2,step=3)
	time3 = time.time()

	print(time1-time0,time2-time1,time3-time2)
	for regco in regCo:
		for k in K:
			a = np.array(train_base[regco][k])
			b = np.array(train_withBias[regco][k])
			c = np.array(train_var_toOne[regco][k])
			xy = np.array([a,b,c])
			#xy = np.asarray([train_base[regco][k],train_withBias[regco][k],train_var_toOne[regco][k]])
			print(xy)
			print(xy.shape)
			colors = ['r','g','b']
			legends = ['base','withBias','with variance']
			plotError(xy, title = 'Train Error of baseline',file_name = 'trains.jpg',colors=colors, legends = legends)


if __name__ == '__main__':
	main()