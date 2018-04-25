import os
import numpy as np
import random
import statistics

class preprocess(object):
	def __init__(self,filepath,density):
		self.filepath = filepath
		self.density = density
		
	def findValid(self):
		#The raw list of all reviews

		f = open(self.filepath,'r')
		rawString = f.read()
		rawList = rawString.split('\n')[:-1]
		for i in range(len(rawList)):
			rawList[i] = rawList[i].split(',')
			rawList[i] = tuple(rawList[i][:-1])

		#Take the first column of rawList (reviewer ID strings)
		reviewerbyreviews = np.array(rawList)[:,0]
		#reviewers: reviewer list without repeat(string); counts: the number of reviews by each person
		reviewers, counts = np.unique(reviewerbyreviews, return_counts=True)
		#{reviewerID(string): pos number of the reviewer}
		reviewersMap = dict(list(zip(reviewers,range(len(reviewers)))))

		#index sequence of large counts, like [[2],[3],..]
		validSeq = np.argwhere(counts>=self.density)

		#filtered raw list(ID string, ID string,rating number)
		validList = [review for review in rawList if [reviewersMap[review[0]]] in validSeq]
		return validList

	def separate(self,validList):
		#Take the first column of validList (reviewer ID strings)
		reviewersbyreviews = np.array(validList)[:,0]
		#reviewer list without repeat(string)
		reviewers = np.unique(reviewersbyreviews)
		#{reviewerID(string): i}
		reviewersMap = dict(list(zip(reviewers,range(len(reviewers)))))

		#Take the second column of validList (item ID strings)
		itemsbyreviews = np.array(validList)[:,1]
		#item list without repeat(string)
		items = np.unique(itemsbyreviews)
		#{itemID(string): j}
		itemsMap = dict(list(zip(items,range(len(items)))))

		#Update the validList to (i, j, rating number) list, according to reviewersMap and itemsMap
		for i in range(len(validList)):
			validList[i] = tuple([reviewersMap[validList[i][0]], itemsMap[validList[i][1]], float(validList[i][2])])
		validSet = set(validList)
		testSet = set()

		for reviewer in reviewers:
			#Retrive all the reviews by the current reviewer
			availableRecords = set(review for review in validList if (review[0] == reviewersMap[reviewer]))
			#Randomly take at least 1/5 of the reviews and put them into testSet
			random_sample_test = set(random.sample(availableRecords,self.density//5+1))
			for i in random_sample_test:
				testSet.add(i)
		#Remaining records go to trainSet
		trainSet = validSet.difference(testSet)
		return validSet, trainSet,testSet, reviewersMap, itemsMap

	def dataProcessing(self):
		validList = self.findValid()
		validSet, trainSet, testSet, user, item = self.separate(validList)
		#reviewersMap -> user; itemsMap -> item
		#validSet, trainSet, testSet are sets of (i, j, rating no) tuples
		#Note that validList with ID strings has been replaced

		trainList = np.array(list(trainSet))
		training_len = len(trainList)
		train_user_id = trainList[:,0]# I-pos no of users in training data with duplicated entries
		train_item_id = trainList[:,1]# J-pos no of items in training data with duplicated entries
		train_rating = trainList[:,2]# corresponding ratings
		return user,item,train_user_id,train_item_id,train_rating,testSet,trainList

	def create_rating_list(self):
		user,item,train_user_id,train_item_id,train_rating,testSet,trainSet=self.dataProcessing()
		#### creating dictionary {user i: array[item j, rating r]}.
		ratings_by_i=dict()
		#### creating dictionary {item j: array[user i, rating r]}.
		ratings_by_j=dict()
		
		for key,value in user.iteritems():#key:IDstring; value:i
			pair = np.argwhere(train_user_id==value)#[[2],[5],..]
			ratings_by_i[value]=np.array([(train_item_id[pair[j][0]],train_rating[pair[j][0]]) for j in range(pair.shape[0])]).astype(int)
		for key,value in item.iteritems():
			pair = np.argwhere(train_item_id==value)
			ratings_by_j[value]=np.array([(train_user_id[pair[k][0]],train_rating[pair[k]][0]) for k in range(pair.shape[0])]).astype(int)
		return ratings_by_i,ratings_by_j