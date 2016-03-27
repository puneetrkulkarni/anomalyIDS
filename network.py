## PK: This module is to implement gradient descent learning algorithm
## for a feedforward network. Gradients are calculated using BP.

import random
import numpy as np
import time as time1
import ConfusionMatrix as cm

class Network(object):
    def __init__(self,sizes):
        ## sizes[] : contains no. of neurons in the respective layers of the network.
        ## Eg: [2,3,1] => first layer contain 2 neurons, second layer contain 3 neurons and 3rd contain 1 neurons.
        ## The biases and weights for the networks are initialized randomly, using gaussian distribution witb mean=0, variane=1
        ## 1st layer -> Input layer, 3rd layer is output layer
        ## biased for neurons are set only for the hidden layer and output layer.
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x)
                    for x,y in zip(sizes[:-1],sizes[1:])]
        self.weights = np.asarray(self.weights)
        #print('weights:',weights_np)
        #print('weights_shape:',weights_np.shape)
        #print('weights_size:',np.array(self.weights.shape).shape)
    def evaluate(self,test_data):
        #forward_res = [self.feedforward(x) for (x,y) in test_data]
        #print('forward_res \t',forward_res)
        test_results = [(np.amax(self.feedforward(x)),(y)) for (x,y) in test_data]
        test_results1 = [((1 if x>0.5 else 0),y) for (x,y) in test_results] 
        #print('test_result:\t',test_results1)
        total = sum(int(x==y) for (x,y) in test_results1)
        tp,fp,tn,fn=cm.ComputeConfusionMatrix(test_results1)
        print('True Positive:',tp)
        print('False Positive:',fp)
        print('True Negative:',tn)
        print('False Negative:',fn)
        return total

    def cost_derivative(self,output_activations,y):
        return (output_activations-y)
    
    def feedforward(self,a):
        ## return the output of the network , calculated as sigmoid(summaation(w.a) + b)
        
        for b,w in zip(self.biases,self.weights):
           # print('b {} \t w {}'.format(b,w))
            a = sigmoid(np.dot(w,a)+b)
            #print('a',a)
            #print('--------------------------------------------')
        
        #print('After activation fun, Y[i] is: ',a)
        return a

    def SGD(self, training_data,epochs,mini_batch_size,eta,test_data):
        ## Train the NN using mini-batch stochastic gradient descent.
        ## The training fata is a list of tuples (x,y) x-> training inputs & y-> desired output
        training_data_list = list(training_data)
        test_data_list = list(test_data)
        
            
        if test_data:
            n_test = len(test_data_list)
        n = len(training_data_list)
        for j in range(epochs):
            random.shuffle(training_data_list)
            #training_data_list = list(training_data)
            mini_batches = [training_data_list[k:k+mini_batch_size]
                            for k in range(0,n,mini_batch_size)]
        
            print('Training Begin at:',time1.ctime())
            t1=time1.time()
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            print('Training End at:',time1.ctime())
            t2=time1.time()
            if test_data:           
                print('Testing Begin at:',time1.ctime())
                t3=time1.time()
                matched_op = self.evaluate(test_data_list)
                print('Testing End at:',time1.ctime())
                t4=time1.time()            
                print("Epoch {0}:{1}/{2}".format(j,matched_op,n_test))
                print('Accuracy(%)',matched_op/n_test*100.0)
                print('Time Taken for Training:',t2-t1)
                print('Time Taken for Testing:',t4-t3)
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self,mini_batch,eta):
        ## update the networks weight and biases by applying gradient descent using backpropogation to a single mini batch
        ## 'mini_batch' is a list of tuples (x,y) & is the learing rate.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            #print('Before:',y)
            #print('After:',y)
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            #print('Modified Biases values:',delta_nabla_b)
            #print('Modified Weight values:',delta_nabla_w)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            #print('Updated Biases :',nabla_b)
            #print('Updated Weights:',nabla_w)
        self.weights = [w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]

    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # FEEDFORWARD Pass
        activation = x
        activations = [x]
        zs = []
        layer=1
        #print('Shape of X:',x.shape)
        for b,w in zip(self.biases,self.weights):
            #print('Layer{} has bias value {} '.format(layer,b))
            #print('Layer{} has Weight matrix value {}'.format(layer,w))
            layer+=1
            sop_weights_inputs = np.dot(w,activation)
            #print('Lets display weights and X')
            #print('X::\t',activation.shape)
            #print('Weights::\t',w.shape)
            
            #print('SOP :\t',sop_weights_inputs)
            #print('SOP Shape :\t',sop_weights_inputs.shape)
            #print('B:\t',b)
            z=sop_weights_inputs+b
            #print('Z value: \t',z)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #print('Activation Array is:\t {}'.format(activations))

        # Backward pass
        delta = self.cost_derivative(activation[-1],y)*sigmoid_prime(zs[-1])
        #print("Delta1:\t",delta)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for bklayer in range(2,self.num_layers):
            z = zs[-bklayer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-bklayer+1].transpose(),delta)*sp
            nabla_b[-bklayer] = delta
            nabla_w[-bklayer] = np.dot(delta,activations[-bklayer-1].transpose())
        return (nabla_b,nabla_w)
   
# misc func
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
