import numpy as np


csv_fpath = 'cancer_data.csv'
df =np.genfromtxt('cancer_data.csv', delimiter=',')


rows = len(df)
cols = len(df[0])

##Q1.a
x=np.ones((rows,cols))
y=np.ones((rows,1))

for i in range(rows):
    for j in range(cols-1):
        x[i][j+1]=df[i][j]
        y[i]=df[i][cols-1]

print("rows : " , rows)
print("cols : " , cols)
print('\n\n')

transposedMat = np.copy(df.T)

for i in range(rows):
    for j in range(cols):
        if np.var(transposedMat[j]) != 0:
            df[i][j]=(df[i][j]-np.average(transposedMat[j]))/np.sqrt(np.var(transposedMat[j]))

print (df)

##Q1.b

def h(x,theta):
    '''
    :param x: a list(i.e. vector) of a row in the X transposed matrix
    :param theta: a list(i.e. vector) of parameters that characterize the model
    :return: the model i.e. the result of the linear function of the model
    '''
    h = 0
    for i in range(len(x)):
        h = h + x[i]*theta[i]
    return h

##Q1.c




