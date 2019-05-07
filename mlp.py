import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np


class NetworkLayer(object):
    def __init__(self,number_inputs, number_neurons):
        self.h = []
        for index in range(len(number_neurons)):
            if(index == 0):
                self.h.append(NeuronLayer(number_inputs=number_inputs,number_neurons=number_neurons[index]))
            else:
                self.h.append(NeuronLayer(number_inputs=number_neurons[index-1],number_neurons=number_neurons[index]))

    def feedforward_layer(self,input):
        for index, y in enumerate(self.h):
            if(index == 0):
                y.feedforward_layer(input)
            else:
                y.feedforward_layer(self.h[index-1].output)
        return y.output[-1:]

    def backward_layer(self,input):
        temp = self.h[::-1]
        for index, y in enumerate(temp):
            if(index == 0):
                y.backpropagation_layer(input)
            else:
                y.backpropagation_h_layer(temp[index-1].update,temp[index-1].weights)

class NeuronLayer(object):
#    input = np.array(1)
#    output = np.array(1)
#    weights = np.array()

    def __init__(self,number_inputs, number_neurons):
        self.alpha = 0.01
        self.input = np.zeros(number_inputs)
        self.output = np.zeros(number_neurons)
        self.update = np.zeros((number_inputs,number_neurons))
        self.weights = np.zeros((number_inputs,number_neurons))
        self.h = [ Neuron(number_inputs=number_inputs,act_funct='relu',alpha=self.alpha) for i in range(number_neurons)]
       
    def feedforward_layer(self,data):
        self.input = data
        for index, y in enumerate(self.h):
            self.weights[:,index]=y.weights
            y.feedforward(self.input)
            self.output[index]=y.output

    def backpropagation_layer(self,delta):
        for index, y in enumerate(self.h):
            self.update[:,index] = y.backpropagation(delta)
            

    def backpropagation_h_layer(self,delta,weights):
        for index, y in enumerate(self.h):
            self.update[:,index] = y.backpropagation(delta[index]*weights[index])

class Neuron(object):
    net = 0
    input = []
    output = 0
    udapte = 0
    def __init__(self, number_inputs, act_funct='relu', reg_lambda=0, bias_flag=False,alpha=0.001):
        '''
            Constructor method. Defines the characteristics of the MLP
        Arguments:
            size_layers : List with the number of Units for:
                [Input, Hidden1, Hidden2, ... HiddenN, Output] Layers.
            act_funtc   : Activation function for all the Units in the MLP
                default = 'sigmoid'
            reg_lambda: Value of the regularization parameter Lambda
                default = 0, i.e. no regularization
            bias: Indicates is the bias element is added for each layer, but the output
     def __init__(self, number_of_neurons, ):
         self.weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1
        '''
        self.act_f          = act_funct
        self.lambda_r       = reg_lambda
        self.bias_flag      = bias_flag
        self.bias           = 0.0
        self.number_inputs  = number_inputs
        #self.weights        = np.ones(self.number_inputs)
        self.weights        = 2 * np.random.random((self.number_inputs)) - 1
        self.output         = 0
        self.alpha          = alpha

    def activation_function(self,x):
        if self.act_f == 'sigmoid':
            g = self.sigmoid(x)
            #g = lambda x: self.sigmoid(x)
        elif self.act_f == 'relu':
            g = self.relu(x)
        elif self.act_f == 'none':
            g = x
            #g = lambda x: self.relu(x)
        return g

    def activation_function_derivative(self,x):
        if self.act_f == 'sigmoid':
            g = self.sigmoid_derivative(x)
            #g = lambda x: self.sigmoid(x)
        elif self.act_f == 'relu':
            g = self.relu_derivative(x)
        elif self.act_f == 'none':
            g = 1
            #g = lambda x: self.relu(x)
        return g

    def feedforward(self,inputs):
        self.input = inputs
        self.net = np.dot(self.weights,self.input) + self.bias
        self.output = self.activation_function(self.net)
        return self.output
  
    def backpropagation(self,delta):
        self.update = delta * self.activation_function_derivative(self.net)
        self.weights = self.weights - self.alpha * self.update * self.input 
        return (np.ones(self.number_inputs)*self.update)

    def sigmoid(self, z):
        '''
        Sigmoid function
        z can be an numpy array or scalar
        '''
        result = 1.0 / (1.0 + np.exp(-z))
        return result

    def relu(self, z):
        '''
        Rectified Linear function
        z can be an numpy array or scalar
        '''
        if np.isscalar(z):
            result = np.max((z, 0))
        else:
            zero_aux = np.zeros(z.shape)
            meta_z = np.stack((z , zero_aux), axis = -1)
            result = np.max(meta_z, axis = -1)
        return result

    def sigmoid_derivative(self, z):
        '''
        Derivative for Sigmoid function
        z can be an numpy array or scalar
        '''
        result = self.sigmoid(z) * (1 - self.sigmoid(z))
        return result

    def relu_derivative(self, z):
        '''
        Derivative for Rectified Linear function
        z can be an numpy array or scalar
        '''
        result = 1 * (z > 0)
        return result



if __name__ == "__main__":
    pass
    in1 = np.array([0,1,0,1])
    in2 = np.array([0,0,1,1])
    ref_out = np.array([0,1,1,1])
    alpha = 0.01
    
    nn = NetworkLayer(number_inputs=2,number_neurons=[2,1])


    for train in range(500000):
       
        for index in range(len(in1)):
            net_input   =   np.array([in1[index], in2[index]])
            act_out     =   nn.feedforward_layer(net_input)
            delta       =   act_out - ref_out[index]
            nn.backward_layer(delta)
            #print("{:6.0f}".format(train),": ",net_input,"{:5.3f}".format(act_out),ref_out[index],"{:5.3f}".format(delta))
            if(np.mod(train,1000)==0):
                for i in range(len(in1)):
                    net_input   =   np.array([in1[i], in2[i]])
                    act_out     =   nn.feedforward_layer(net_input)
                    delta       =   act_out - ref_out[i]
                    #print(net_input,,"{:5.3f}".format(act_out),delta)                    
                    print("{:6.0f}".format(int(train)),": ",net_input,"{:5.3f}".format(int(act_out)),ref_out[index],"{:5.3f}".format(int(delta)))
#                print("{:6.0f}".format(train),": ",net_input,"{:5.3f}".format(act_out),ref_out[index],"{:5.3f}".format(delta))

    # create plot
    #plt = pg.plot()
    #plt.showGrid(x=True,y=True)
    #plt.addLegend()
    # set properties
    #plt.setLabel('left', 'Value', units='V')
    #plt.setLabel('bottom', 'Time', units='s')
    #plt.setXRange(0,10)
    #plt.setYRange(0,20)
    #plt.setWindowTitle("Transmitted Symbols")
    # plot
    #plt.plot(dataSymbolsOut.real[-300:],dataSymbolsOut.imag[-300:], pen=None, symbol='x', symbolPen='r', symbolBrush=0.2, name='blue')
    #i=range(len(weights))
    #plt.plot(i,weights, pen=None, symbol='o', symbolPen='b', symbolBrush=0.2, name='red')
    #plt.showGrid(x=True,y=True)

        