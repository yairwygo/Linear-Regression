import numpy as np
import matplotlib.pyplot as plt
csv_fpath = 'cancer_data.csv'
matrix =np.genfromtxt('cancer_data.csv', delimiter=',')


rows = len(matrix)
cols = len(matrix[0])

##Q1.a
x=np.ones((rows,cols))
y=np.ones((rows,1))



print("rows : " , rows)
print("cols : " , cols)
print('\n\n')

transposedMat = np.copy(matrix.T)

for i in range(rows):
    for j in range(cols):
        if np.var(transposedMat[j]) != 0:
            matrix[i][j]=(matrix[i][j]-np.average(transposedMat[j]))/np.sqrt(np.var(transposedMat[j]))


for i in range(rows):
    for j in range(cols-1):
        x[i][j+1]=matrix[i][j]
        y[i]=matrix[i][cols-1]
print (x)
print (y)

##Q1.b

def h(xv,theta):
    '''
    :param x: a list(i.e. vector) of a row in the X transposed matrix
    :param theta: a list(i.e. vector) of parameters that characterize the model
    :return: the model i.e. the result of the linear function of the model
    '''
    h = 0.
    for i in range(len(theta)):
        h = h + xv[i]*theta[i]
    return h

##Q1.c
def jErr(x,y,theta):
    jCount=0.
    for i in range(rows):
        jCount = (h(x[i], theta)-y[i])**2

    return jCount/(rows*2)

##Q1.d

def gradient (x,y,theta):
    gradientList=list(0 for i in range(cols))
    #j is the property
    for j in range(cols):
        #i is the example
        for i in range(rows):
            gradientList[j]+=(h(x[i],theta)-y[i])*x[i][j]
        gradientList[j]= gradientList[j]/rows
    return gradientList

##Q1.e

def gradienctDesent(x,y,alpha):

    k=0
    numOfIteration = 20
    epsilon = 0.00001
    delta = 0.00001
    theta0 = np.ones((cols,1))
    theta1 = np.ones((cols,1))
    JList = []

    while True:
        grad = gradient(x,y,theta0)
        for j in range(cols):
            theta1[j]=theta0[j]-alpha*grad[j]

        JList.append(jErr(x,y,theta0))
        if(k > numOfIteration or np.linalg.norm(theta1-theta0) < epsilon  or np.absolute(jErr(x,y,theta1)-jErr(x,y,theta0)) < delta ):
            break
        theta0 = np.copy(theta1)
        k = k + 1

    return JList

plt.plot(gradienctDesent(x,y,0.1), color = "b")
plt.show()
##Q1.f

def miniBatch(x,y,alpha,n):
    k = 0
    numOfIteration = 100
    epsilon = 0.00001
    delta = 0.00001
    theta0 = list(0 for i in range(cols))
    theta1 = list(0 for i in range(cols))
    JList = []
    while True:
        for j in range(cols):
            for i in range(k*n,(k+1*(n-1))):
                theta1[j] = theta0[j] - alpha * gradient(x, y, theta0)
            JList.append(jErr(x, y, theta0))
            if (k > numOfIteration or np.linalg.norm(theta1 - theta0) < epsilon or np.absolute(jErr(x, y, theta1) - jErr(x,y,theta0)) < delta):
                break
            theta0 = np.copy(theta1)
            k = k + 1
    return JList




