import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np


class NetworkLayer(object):
    def __init__(self,number_inputs, number_neurons,number_layer,number_outputs):
        for index in range(number_layer):
            if(index == 0):
                self.h = np.array(number_neurons[0],dtype=Neuron(number_inputs=number_inputs,act_funct='relu',alpha=alpha))
            else:
                self.h.append(Neuron(number_inputs=number_neurons[index-1],act_funct='relu',alpha=alpha))

    def feedforward_layer(self,input):
        for index, y in enumerate(self.h):
            if(index == 0):
                y.feedforward(in)
            else:
                y.feedforward(self.h[index-1].output)
        return x

class NeuronLayer(object):
    input = np.array()
    output = np.array()
#    weights = np.array()

    def __init__(self,number_inputs, number_neurons):
        alpha = 0.01
        input = np.array(number_inputs)
        output = np.array(number_neurons)
#        weights = np.array()
        self.h = np.array(number_neurons,dtype=Neuron(number_inputs=number_inputs,act_funct='relu',alpha=alpha))

    def feedforward_layer(self,data):
        input = data
        for index, y in enumerate(self.h):
            y.feedforward(data)
            output[index]=y.output

    def backpropagation_layer(self,delta):
        for index, y in enumerate(self.h):
            y.backpropagation(delta)


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
        self.bias           = 0.5
        self.number_inputs  = number_inputs
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
        return self.weights

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
    ref_out = np.array([0,0,0,1])
    alpha = 0.01


        
    out_layer  = Neuron(number_inputs=5,act_funct='sigmoid')
    
    weights=[]
    error=[]
    for train in range(500000):
        
        #weights.append([h1.weights[0],h1.weights[1],h2.weights[0],h2.weights[1],out_layer.weights[0],out_layer.weights[1]])
        for index in range(len(in1)):
            net_input   =   [in1[index], in2[index]]

            act_vh1     =   h1.feedforward(np.array(net_input))
            act_vh2     =   h2.feedforward(np.array(net_input))
            act_vh3     =   h3.feedforward(np.array(net_input))
            act_vh4     =   h4.feedforward(np.array(net_input))
            act_vh5     =   h5.feedforward(np.array(net_input))     

            act_out     =   out_layer.feedforward(np.array([act_vh1,act_vh2,act_vh3,act_vh4,act_vh5]))
            
            delta       =   act_out - ref_out[index]
            out_layer.backpropagation(delta)
            h1.backpropagation(out_layer.weights[0]*out_layer.update)
            h2.backpropagation(out_layer.weights[1]*out_layer.update)
            h3.backpropagation(out_layer.weights[2]*out_layer.update)
            h4.backpropagation(out_layer.weights[3]*out_layer.update)
            h5.backpropagation(out_layer.weights[4]*out_layer.update)
            if(np.mod(train,20000)==0):
                print("{:6.0f}".format(train),": ",net_input,"{:5.3f}".format(act_out),ref_out[index],"{:5.3f}".format(delta))

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

        