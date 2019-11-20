import numpy as np
import matplotlib.pyplot as plt
matrix =np.genfromtxt('cancer_data.csv', delimiter=',')
rows = len(matrix)
cols = len(matrix[0])

##Q1.a
x=np.ones((rows,cols))
y=np.ones((rows,1))


transposedMat = np.copy(matrix.T)

for i in range(rows):
    for j in range(cols):
        if np.var(transposedMat[j]) != 0:
            matrix[i][j]=(matrix[i][j]-np.average(transposedMat[j]))/np.sqrt(np.var(transposedMat[j]))


for i in range(rows):
    for j in range(cols-1):
        x[i][j+1]=matrix[i][j]
    y[i]=matrix[i][cols-1]

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
        jCount += (h(x[i], theta)-y[i])**2

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

def gradientDescent(x,y,alpha):

    k=0
    numOfIteration = 20
    epsilon = 0.001
    delta = 0.001
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
plt.figure("Gradient Descent - different alphas")
plt.plot(gradientDescent(x,y,0.1), color = "b")
plt.plot(gradientDescent(x,y,0.01), color = "r")
plt.plot(gradientDescent(x,y,0.001), color = "g")
plt.show()

##Q1.f

def miniBatch(x,y,alpha,n):
    k = 0
    numOfIteration = 100
    epsilon = 0.001
    delta = 0.001
    theta0 = np.ones((cols, 1))
    theta1 = np.ones((cols, 1))
    JList = []
    while True:
        for j in range(cols):
            for i in range(k* n, (k+1)*n-1):
                new_i = i % rows
                theta1[j] = theta1[j] - (alpha * ((h(x[new_i],theta0)-y[new_i])*x[new_i][j]))/n
        JList.append(jErr(x, y, theta0))
        if (k > numOfIteration or np.linalg.norm(theta1 - theta0) < epsilon or np.absolute(jErr(x, y, theta1) - jErr(x,y,theta0)) < delta):
            break
        theta0 = np.copy(theta1)
        k = k + 1
    return JList


plt.figure("Mini Batch")
plt.plot(miniBatch(x,y,0.1,50),color="b")
plt.title("Mini Batch")
plt.show()

#Q1.g
def momentum (x,y):
    v=np.zeros((cols,1))
    v1=np.copy(v)
    numOfIteration = 100
    epsilon = 0.001
    delta = 0.001
    theta0=np.ones((cols,1))
    theta1=np.copy(theta0)
    JList = []
    k=0

    while True:
        Err=gradient(x,y,theta0)

        for j in range(cols):
                v1[j]=0.9*v[j]+0.1*Err[j]
                theta1[j]=theta0[j]-v1[j]

        JList.append(jErr(x, y, theta0))
        if (k > numOfIteration or np.linalg.norm(theta1 - theta0) < epsilon or np.absolute(jErr(x, y, theta1) - jErr(x,y,theta0)) < delta):
            break
        theta0 = np.copy(theta1)
        v = np.copy(v1)
        k = k + 1
    return JList

plt.figure("Momentum")
plt.plot(momentum(x,y),color="b")
plt.title("Momentum")
plt.show()





