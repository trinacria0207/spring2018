import numpy as np 
import pandas as pd 

covert npy to readable csv file
a = np.load('ecs171train.npy')
b = np.load('ecs171test.npy')
l1 = []
for i in range(0, len(a)):
    lt = a[i].decode('ASCII').strip().split(',')
    l1.append(lt)
l1 = pd.DataFrame.from_records(l1[1:], columns=l1[:1])
l1.to_csv("train.csv", sep=',',  index=False)

l2 = []
for i in range(0, len(b)):
    lt = b[i].decode('ASCII').strip().split(',')
    l2.append(lt)
l2 = pd.DataFrame.from_records(l2, columns=l1.columns.values[:-1])
l2.to_csv("test.csv", sep=',',  index=False)

#read the data

train = pd.read_csv("./train.csv",low_memory=False)
print("read train")
test = pd.read_csv("./test.csv",low_memory=False)
print("read test")
train = train.fillna(train.median())
test = test.fillna(test.median())
# features we finally selected
pair = [['f274','f528'],['f274','f527'],['f267','f578'],['f265','f578'],['f527','f528'],['f264','f578'],['f538','f539'],['f266','f578']]
unit = ['id','f663','f421','f221','f776','f335','f292','f219','f428','f674','f415','f251','f766','f414','f314','f536','f404','f526']
unitr = ['f663','f421','f221','f776','f335','f292','f219','f428','f674','f415','f251','f766','f414','f314','f536','f404','f526','loss']

t = test[unit]
for i in pair:
    a,b = i
    t[a+"-"+b] = test[a] - test[b]
t.to_csv("test_2.csv", sep=',',  index=False)

tr = train[unitr]
for i in pair:
    a,b = i
    tr[a+"-"+b] = train[a] - train[b]
    
tr.to_csv("submission.csv", sep=',',  index=False)

#get new test data with selected features

