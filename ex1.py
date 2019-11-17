import numpy as np
import pandas as pd
import matplotlib as plt
import pandas

csv_fpath = '/Users/yairwygoda/Desktop/Machine_Learning_ex1/cancer_data.csv'
df = pandas.read_csv(csv_fpath ,names=['x0', 'x1','x2', 'x3', 'x4','x5', 'x6', 'x7','x8', 'y'])
matrix = df.values.tolist()
testMat = df.values.tolist()
dimentions = np.shape(matrix)
print("the matrix dimentions : " , dimentions)
rows = dimentions[0]
cols = dimentions[1]
print("rows : " , rows)
print("cols : " , cols)
print('\n\n')

#turn list into matrix
matrix = np.asmatrix(matrix)
transposedMat = matrix.transpose()
testMat = np.asmatrix(testMat)
testMatTrans= testMat.transpose()
print('transposedMat before normalizing : \n',transposedMat)
print('\n\n')

#computing means and standard diviations of every row in the transposed matrix
meansList = list()
standardDeviationList = list()
for i in transposedMat:
    meansList.append(np.mean(i))
    standardDeviationList.append(np.std(i))

#normalize the transposed matrix
for i in  range(cols):
    #print('\n',meansList[i])
    for j in range(rows):
        normalized = (transposedMat.item(i, j) - meansList[i])/standardDeviationList[i]
        transposedMat.itemset((i, j), normalized)
        #print(transposedMat.item(i, j))


#print('\n\n\n\n')
print('\nafter normalizing')
print(transposedMat)


means= list()
standardDeviations = list()
for i in transposedMat:
    means.append(np.mean(i))
    standardDeviations.append(np.std(i))


print('means ave : ',np.mean(means))
print('standardDeviations ave : ',np.mean(standardDeviations))

print('\n\n')


