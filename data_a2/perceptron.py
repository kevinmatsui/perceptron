import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
np.random.seed(1)



def perceptronAvg(X,lr,epochs, weird):
    numExamples, numFeatures = X.shape
    ones = np.ones(numExamples)
    development = np.loadtxt("developmental_data.csv", delimiter=",")
    # initialize w
    a = np.random.uniform(-0.01,0.01,(1,numFeatures))
    w = np.random.uniform(-0.01,0.01,(1,numFeatures))
    # list stores number of misses for each epoch
    number_misses = []
    wVectors = []
    accuracies = []
    miss = 0
    w = w.reshape(-1)
    a = a.reshape(-1)
    copy = X

    reshaped = X[0].reshape(-1,1)
    for epoch in range(epochs):
        X = copy   
        np.random.shuffle(X)
        y = X[:,0]
        y[y ==0] = -1
        X = np.delete(X,0,1)
        X = np.insert(X, 0, ones, axis = 1)
        for i in range(len(X)):
            #val = X[i].reshape(-1,1)
            if y[i]*np.dot(w.T,X[i]) < 0:
                w = w + np.multiply(X[i],lr*y[i])
                miss+=1
        a = a + w
        a = a/(epoch+1)
        number_misses.append(miss)
        if weird:
            accuracies.append(test_w(development, a))
            wVectors.append(a)
    
    if weird:
        maxIndex = np.argmax(accuracies)
        return wVectors[maxIndex], accuracies,miss
    return a



def perceptronMargin(X,lr,epochs,margin, weird):
    numExamples, numFeatures = X.shape
    ones = np.ones(numExamples)
    development = np.loadtxt("developmental_data.csv", delimiter=",")
    # initialize w
    w = np.random.uniform(-0.01,0.01,(1,numFeatures))
    # list stores number of misses for each epoch
    number_misses = []
    wVectors = []
    accuracies = []
    miss = 0
    w = w.reshape(-1)
    copy = X

    reshaped = X[0].reshape(-1,1)
    for epoch in range(epochs):
        dlr = lr/(1+epoch)
        X = copy   
        np.random.shuffle(X)
        y = X[:,0]
        y[y ==0] = -1
        X = np.delete(X,0,1)
        X = np.insert(X, 0, ones, axis = 1)
        for i in range(len(X)):
            #val = X[i].reshape(-1,1)
            if y[i]*np.dot(w.T,X[i]) < margin:
                w = w + np.multiply(X[i],dlr*y[i])
                miss+=1
        number_misses.append(miss)
        if weird:
            accuracies.append(test_w(development, w))
            wVectors.append(w)
    
    if weird:
        maxIndex = np.argmax(accuracies)
        return wVectors[maxIndex], accuracies,miss
    return w



def perceptronDecay(X,lr,epochs, weird):
    numExamples, numFeatures = X.shape
    ones = np.ones(numExamples)
    development = np.loadtxt("developmental_data.csv", delimiter=",")
    # initialize w
    w = np.random.uniform(-0.01,0.01,(1,numFeatures))
    # list stores number of misses for each epoch
    number_misses = []
    wVectors = []
    accuracies = []
    miss = 0
    w = w.reshape(-1)
    copy = X

    reshaped = X[0].reshape(-1,1)
    for epoch in range(epochs):
        dlr = lr/(1+epoch)
        X = copy   
        np.random.shuffle(X)
        y = X[:,0]
        y[y ==0] = -1
        X = np.delete(X,0,1)
        X = np.insert(X, 0, ones, axis = 1)
        for i in range(len(X)):
            #val = X[i].reshape(-1,1)
            if y[i]*np.dot(w.T,X[i]) < 0:
                w = w + np.multiply(X[i],dlr*y[i])
                miss+=1
        number_misses.append(miss)
        if weird:
            accuracies.append(test_w(development, w))
            wVectors.append(w)
    
    if weird:
        maxIndex = np.argmax(accuracies)
        return wVectors[maxIndex], accuracies,miss
    return w

def perceptron(X,lr,epochs, weird):
    numExamples, numFeatures = X.shape
    ones = np.ones(numExamples)
    development = np.loadtxt("developmental_data.csv", delimiter=",")
    # initialize w
    w = np.random.uniform(-0.01,0.01,(1,numFeatures))
    # list stores number of misses for each epoch
    number_misses = []
    wVectors = []
    accuracies = []
    miss = 0
    w = w.reshape(-1)
    copy = X

    reshaped = X[0].reshape(-1,1)
    for epoch in range(epochs):
        X = copy   
        np.random.shuffle(X)
        y = X[:,0]
        y[y ==0] = -1
        X = np.delete(X,0,1)
        X = np.insert(X, 0, ones, axis = 1)
        for i in range(len(X)):
            #val = X[i].reshape(-1,1)
            if y[i]*np.dot(w.T,X[i]) < 0:
                w = w + np.multiply(X[i],lr*y[i])
                miss+=1
        number_misses.append(miss)
        if weird:
            accuracies.append(test_w(development, w))
            wVectors.append(w)
    
    if weird:
        maxIndex = np.argmax(accuracies)
        return wVectors[maxIndex], accuracies,miss
    return w

def test_w(X, w):
    numExamples, numFeatures = X.shape
    y = X[:,0]
    y[y ==0] = -1
    ones = np.ones(numExamples)
    X = np.delete(X,0,1)
    X = np.insert(X, 0, ones, axis = 1)
    mistakes = 0
    for x in range(len(X)):
        if y[x]*np.dot(w,X[x]) < 0:
            mistakes+=1
    accuracy = (numExamples-mistakes)/numExamples
    return accuracy

#data = np.loadtxt("/CVSplits/training", delimiter=",")

def cross_valid(variation,epochs, lr, margin):
    if variation == 0 or variation == 1 or variation == 3:
        print("Cross Validation For LR Value: " + str(lr))
    else:
        print("Cross Validation For LR Value: " + str(lr))
        print("Cross Validation For Margin Value: " + str(margin))
    
    fold1 = np.loadtxt("training0.csv", delimiter=",")
    fold2 = np.loadtxt("training1.csv", delimiter=",")
    fold3 = np.loadtxt("training2.csv", delimiter=",")
    fold4 = np.loadtxt("training3.csv", delimiter=",")
    fold5 = np.loadtxt("training4.csv", delimiter=",")
    # y1 = fold1[:,0]
    # fold1 = np.delete(fold1,0,1)
    # print("test removing labels:")
    # print("training 0 labels:")
    # print(y1)
    # print("training 0 without labels:")
    # print(fold1)
    # y2 = fold2[:,0]
    # fold2 = np.delete(fold2,0,1)
    # y3 = fold3[:,0]
    # fold3 = np.delete(fold3,0,1)
    # y4 = fold4[:,0]
    # fold4 = np.delete(fold4,0,1)
    # y5 = fold5[:,0]
    # fold5 = np.delete(fold5,0,1)
    accuracies =[]
    for x in range(5):
        if x == 0:
            train = np.concatenate([fold2,fold3,fold4,fold5], axis = 0)
            test = fold1
        if x == 1:
            train = np.concatenate([fold3,fold4,fold5,fold1], axis=0)
            #labels = np.concatenate([y3,y4,y5,y1], axis = 0)
            test = fold2
        if x == 2:
            train = np.concatenate([fold4,fold5,fold1, fold2], axis=0)
            #labels = np.concatenate([y4,y5,y1,y2], axis = 0)
            test = fold3
        if x == 3:
            train = np.concatenate([fold5,fold1,fold2,fold3], axis=0)
            #labels = np.concatenate([y5,y1,y2,y3], axis = 0)
            test = fold4
        if x == 4:
            train = np.concatenate([fold1,fold2,fold3,fold4], axis=0)
            #labels = np.concatenate([y1,y2,y3,y4], axis = 0)
            test = fold5
        if variation ==0:
            wVec = perceptron(train,lr,epochs,False)
            tester = test_w(test,wVec)
            accuracies.append(tester)
            print("test accuracy fold" + str(x+1) + ": " + str(tester))
            
        if variation ==1:
            wVec = perceptronDecay(train,lr,epochs,False)
            tester = test_w(test,wVec)
            accuracies.append(tester)
            print("test accuracy fold" + str(x+1) + ": " + str(tester))
        if variation ==2:
            wVec = perceptronMargin(train,lr,epochs,margin,False)
            tester = test_w(test,wVec)
            accuracies.append(tester)
            print("test accuracy fold" + str(x+1) + ": " + str(tester))
        if variation == 3:
            wVec = perceptronAvg(train,lr,epochs,False)
            tester = test_w(test,wVec)
            accuracies.append(tester)
            print("test accuracy fold" + str(x+1) + ": " + str(tester))
    sum = 0
    for x in accuracies:
        sum+=x
    avg = sum/len(accuracies)
    print("Averaged test accuracy: " + str(avg))
    return avg


training = np.loadtxt("training_data.csv", delimiter=",")
testing = np.loadtxt("testing_data.csv", delimiter=",")
develop = np.loadtxt("developmental_data.csv", delimiter=",")
print("REGULAR PERCEPTRON:")
reg_percep_accur = []
reg_percep_accur.append(cross_valid(0,10,1,0))
reg_percep_accur.append(cross_valid(0,10,.1,0))
reg_percep_accur.append(cross_valid(0,10,.01,0))
lr_hyper = [1,.1,.01]
time.sleep(0.5)
print("")
print("The best hyperparameters: ")
print("Learning Rate: " + str(lr_hyper[np.argmax(reg_percep_accur)]))
print("Cross-validation accuracy for best hyperparameter: " + str(np.max(reg_percep_accur)))
tupleVal = perceptron(training, lr_hyper[np.argmax(reg_percep_accur)],20, True)
print("Total number of updates learning algorithm performs on training set: ", str(tupleVal[2]))
print("Development Set Accuracy for each epoch: ")
print(tupleVal[1])
print("Best Epoch Accuracy for Development Set:" + str(np.max(tupleVal[1])))
print("Test Set Accuracy: " + str(test_w(testing,tupleVal[0])))
xaxis = list(range(1,21))
plt.plot(xaxis, tupleVal[1])
  
# naming the x axis
plt.xlabel('epoch id')
# naming the y axis
plt.ylabel('dev set accuracy')
  
# giving a title to my graph
plt.title('Reg Perceptron Learning Curve')
  
# function to show the plot
plt.savefig("regular.png")
plt.clf()
print(" ")

print("PERCEPTRON DECAYING LEARNING RATE:")
dec_percep_accur = []
dec_percep_accur.append(cross_valid(1,10,1,0))
dec_percep_accur.append(cross_valid(1,10,.1,0))
dec_percep_accur.append(cross_valid(1,10,.01,0))
lr_hyper = [1,.1,.01]
time.sleep(0.5)
print("")
print("The best hyperparameters: ")
print("Learning Rate: " + str(lr_hyper[np.argmax(dec_percep_accur)]))
print("Cross-validation accuracy for best hyperparameter: " + str(np.max(dec_percep_accur)))
tupleVal = perceptronDecay(training, lr_hyper[np.argmax(dec_percep_accur)],20, True)
print("Total number of updates learning algorithm performs on training set: ", str(tupleVal[2]))
print("Development Set Accuracy for each epoch: ")
print(tupleVal[1])
print("Best Epoch Accuracy for Development Set:" + str(np.max(tupleVal[1])))
print("Test Set Accuracy: " + str(test_w(testing,tupleVal[0])))
xaxis = list(range(1,21))
plt.plot(xaxis, tupleVal[1])
  
# naming the x axis
plt.xlabel('epoch id')
# naming the y axis
plt.ylabel('dev set accuracy')
  
# giving a title to my graph
plt.title('Perceptron Decaying Learning Rate Learning Curve')
  
# function to show the plot
plt.savefig("decaying.png")
plt.clf()
print(" ")


print("MARGIN PERCEPTRON:")
mar_percep_accur = []
mar_percep_accur.append(cross_valid(2,10,1,1))
mar_percep_accur.append(cross_valid(2,10,.1,1))
mar_percep_accur.append(cross_valid(2,10,.01,1))
mar_percep_accur.append(cross_valid(2,10,1,.1))
mar_percep_accur.append(cross_valid(2,10,.1,.1))
mar_percep_accur.append(cross_valid(2,10,.01,.1))
mar_percep_accur.append(cross_valid(2,10,1,.01))
mar_percep_accur.append(cross_valid(2,10,.1,.01))
mar_percep_accur.append(cross_valid(2,10,.01,.01))
time.sleep(0.5)

lr_hyper = [1,.1,.01,1,.1,.01,1,.1,.01]
mar_hyper = [1,1,1,.1,.1,.1,.01,.01,.01]
print("")
print("The best hyperparameters: ")
print("Learning Rate: " + str(lr_hyper[np.argmax(mar_percep_accur)]))
print("Margin: " + str(mar_hyper[np.argmax(mar_percep_accur)]))
print("Cross-validation accuracy for best hyperparameter: " + str(np.max(mar_percep_accur)))
tupleVal = perceptronMargin(training, lr_hyper[np.argmax(mar_percep_accur)],20, mar_hyper[np.argmax(mar_percep_accur)], True)
print("Total number of updates learning algorithm performs on training set: ", str(tupleVal[2]))
print("Development Set Accuracy for each epoch: ")
print(tupleVal[1])
print("Best Epoch Accuracy for Development Set:" + str(np.max(tupleVal[1])))
print("Test Set Accuracy: " + str(test_w(testing,tupleVal[0])))
xaxis = list(range(1,21))
plt.plot(xaxis, tupleVal[1])
  
# naming the x axis
plt.xlabel('epoch id')
# naming the y axis
plt.ylabel('dev set accuracy')
  
# giving a title to my graph
plt.title('Margin Perceptron Learning Curve')
  
# function to show the plot
plt.savefig("margin.png")
plt.clf()
print(" ")

print("AVERAGE PERCEPTRON:")
avg_percep_accur = []
avg_percep_accur.append(cross_valid(3,10,1,0))
avg_percep_accur.append(cross_valid(3,10,.1,0))
avg_percep_accur.append(cross_valid(3,10,.01,0))
lr_hyper = [1,.1,.01]
time.sleep(0.5)
print("")
print("The best hyperparameters: ")
print("Learning Rate: " + str(lr_hyper[np.argmax(avg_percep_accur)]))
print("Cross-validation accuracy for best hyperparameter: " + str(np.max(avg_percep_accur)))
tupleVal = perceptronAvg(training, lr_hyper[np.argmax(avg_percep_accur)],20, True)
print("Total number of updates learning algorithm performs on training set: ", str(tupleVal[2]))
print("Development Set Accuracy for each epoch: ")
print(tupleVal[1])
print("Best Epoch Accuracy for Development Set:" + str(np.max(tupleVal[1])))
print("Test Set Accuracy: " + str(test_w(testing,tupleVal[0])))
xaxis = list(range(1,21))
plt.plot(xaxis, tupleVal[1])
  
# naming the x axis
plt.xlabel('epoch id')
# naming the y axis
plt.ylabel('dev set accuracy')
  
# giving a title to my graph
plt.title('Average Perceptron Learning Curve')
  
# function to show the plot
plt.savefig("average.png")

zCount = 0
oCount = 0
yTest = testing[:,0]
for val in yTest:
    if val == -1:
        zCount +=1
    if val == 1:
        oCount +=1



yTest = develop[:,0]
zCount = 0
oCount = 0
for val in yTest:
    if val == 0:
        zCount +=1
    if val == 1:
        oCount +=1

