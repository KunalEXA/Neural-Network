%matplotlib notebook
import random
from numpy import array, zeros, shape, dot, transpose, asarray
from pylab import plot, ylim
import math
#This is the answer to the second question
#Initilizing values for circle
a = 0.5
b = 0.6
r = 0.4
epochs = 100
stepSize = 0.1

train = []
test = []

#To determine output value for randomly created samples
def computeY(x, y):
    if((x-a)*(x-a)+(y-b)*(y-b) < r*r):
        return 1
    else:
        return 0

#For non-linear transformation
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#Generating 100 random points for training
for _ in range(100):
    x = random.uniform(0,1)
    y = random.uniform(0, 1)
    Y = computeY(x, y)
    temp = [1, x, y]
    newList = [temp, Y]
    train.append(newList)

#Simple feedforward and backpropagation
W1 = zeros(shape=(11,3))
W2 = zeros(shape=(11, 1))
o1 = zeros(shape=(11, 1))
o2 = 0;
 
#Initializing value of bias

for _ in range(epochs):
    for _ in range(100):
        x, y = random.choice(train)

        net1 = dot(W1, x)
        for i in range(11):
            o1[i] = sigmoid(net1[i])
        net2 = dot(transpose(W2), o1)
        o2 = sigmoid(net2);
        
        #Here starts backpropagation
        delta3 = o2*(1 - o2)*(o2-y)
        delta2 = o1*(1 - o1)*(delta3)*W2
        W2 += (-stepSize)*o1*delta3
        W1 += (-stepSize)*delta2*x

#Generating 100 random points for testing
for _ in range(100):
    x = random.uniform(0,1)
    y = random.uniform(0,1)
    Y = computeY(x, y)
    temp = [1, x, y]
    newList = [temp, Y]
    test.append(newList)

accuracy = 0
for i in range(100):
    x = test[i][0]
    y = test[i][1]
    net1 = dot(W1, x)
    for i in range(11):
        o1[i] = sigmoid(net1[i])
    net2 = dot(transpose(W2), o1)
    o2 = sigmoid(net2);
    errors.append(0.5*(y - o2)*(y-o2))
    if(o2 >= 0.5):
        o2 = 1
    else:
        o2 = 0
    if(y == o2):
        accuracy += 1

print(accuracy)
plot(errors)