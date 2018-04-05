import os
import numpy as np
import random
import statistics

def dataToArray():
    path = os.getcwd()
    file = path + '/ratings_Musical_Instruments.csv'
    f = open(file,'r')
    rawString = f.read()
    rawList = rawString.split('\n')[:-1]
    user = dict()
    item = dict()
    user_id = 0
    item_id = 0
    train = set(rawList[x] for x in random_row)
    test_data = list(set(rawList) - train)
    #print type(test)
    #print len(random_row)
    # train__id is the id from training data, they are duplicated entries, 
    # one user can rate multiple items
    # user maps user string to number
    # item_id contains unique item's id
    training_len = len(random_row)
    train_user_id=np.zeros(training_len)
    train_item_id=np.zeros(training_len)
    train_rating=np.zeros(training_len)
    #print len(train_user_id)
    #for i in random_row:
    for row in range(len(rawList)):
        rawList[row] = rawList[row].split(',')
        rawList[row] = tuple(rawList[row][:-1])

        if rawList[row][0] not in user:
            user_id+=1
            user[rawList[row][0]] = user_id
        if rawList[row][1] not in item:
            item_id+=1
            item[rawList[row][1]] = item_id

    for i in range(training_len):
        row = random_row[i]
        train_user_id[i] = user[rawList[row][0]]
        train_item_id[i] = item[rawList[row][1]]
        train_rating[i] = rawList[row][2]


    return user,item,train_user_id,train_item_id,train_rating,test_data