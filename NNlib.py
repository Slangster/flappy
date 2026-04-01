import scipy.special
import numpy
import matplotlib.pyplot
import matplotlib.image
import pickle
import os
import copy
from time import time

class neuralNetwork:
    def __init__(self,structure,learning_rate):
        self.structure=structure
        self.lr =learning_rate
        self.W=[numpy.random.normal(0,pow(structure[k+1],-0.5),(structure[k+1],structure[k])) for k in range(len(structure)-1)]
        #self.activation_function= lambda x: scipy.special.expit(x)  #logistic function, sigmoid   WARNING : should not be used here if Pickled

    def activation_function(self,x):
        return scipy.special.expit(x)  #logistic function, sigmoid

    def query(self,inputs):
        state=numpy.array(inputs,ndmin=2).T
        for k in range(len(self.structure)-1):
            state=self.activation_function(numpy.dot(self.W[k],state))
        return state

    def train(self,inputs,target):
        states=[numpy.array(inputs,ndmin=2).T]
        for k in range(len(self.structure)-1):
            states+=[self.activation_function(numpy.dot(self.W[k],states[k]))]
        
        errors=[target-states[len(self.structure)-1]]
        for k in range(len(self.structure)-2):
            errors+=[numpy.dot(self.W[len(self.structure)-2-k].T,errors[k])]   #errors are listed backwards, from output to input
        
        ###UPDATE###
        for k in range(len(self.structure)-1):
            self.W[k]+=self.lr*numpy.dot((errors[len(self.structure)-2-k]*states[k+1]*(1-states[k+1])),numpy.transpose(states[k]))

    def mutate(self,p,change):
        for k in range(len(self.structure)-1):
            T=self.W[k]
            for i in range(len(T)):
                for j in range(len(T[0])):
                    test=numpy.random.uniform(0,1)
                    if(test<p):
                        T[i][j]+=numpy.random.normal(0,change)
            self.W[k]=T


def save(NN,name,origin):
    with open(origin+name,'wb') as file:
        data_pickler=pickle.Pickler(file)
        data_pickler.dump(NN)

def load(name,origin):
    with open(origin+name,'rb') as file:
        data_unpickler=pickle.Unpickler(file)
        NN=data_unpickler.load()
    return NN
