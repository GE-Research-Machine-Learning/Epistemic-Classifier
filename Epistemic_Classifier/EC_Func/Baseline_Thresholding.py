from keras import backend as K
import numpy as np

def get_activation(x, model, layer_select, verbose=1):
    inp = model.input  # input placeholder
    outputs = []
    if verbose == 1:
        print('layer selected: ')
    for i in layer_select:
        outputs.append(model.layers[i].output)
        if verbose == 1:
            print(model.layers[i].name)

    eval_function = K.function([inp, K.learning_phase()], outputs)  # evaluation function
    result = eval_function([x, 0.])
    data_all = []
    for activation in result:
        res = np.reshape(activation, (x.shape[0], -1))
        # print(res.shape)
        data_all.append(res)

    return data_all

class baseline:
    def __init__(self, model, params, selected_layer=[-1], method='WaldSPRT', numClasses = 2):
        self.model = model
        self.method = method
        self.selected_layer = selected_layer #last layer
        self.input_params = params
        self.model_params = None
        self.numClasses = numClasses

    def fit(self): #, train_x, train_y
        print('Using method:', self.method)
        if self.method == 'WaldSPRT':
            # a\approx \log {\frac  {\beta }{1-\alpha }} and b\approx \log {\frac  {1-\beta }{\alpha }}
            # param[0] is false positive rate
            # param[1] is false negative rate
            a = self.input_params[0]/(1-self.input_params[1])
            b = (1-self.input_params[0])/self.input_params[1]
            self.model_params = [a, b]
        
        if self.method == 'FixedSoftmaxThreshold':
            self.model_params = self.input_params[0]
            
#         if self.method == 'PerClassSoftmaxThreshold':
            #TODO
            
    def predict_class(self, x):
        if self.method == 'WaldSPRT':
            py = get_activation(x, self.model, layer_select=[-1], verbose=1)
            LR = np.divide(py[0][:,0],py[0][:,1]).reshape(-1)
            byhat = np.ones((py[0].shape[0], 1))
            byhat[LR >= 0.5] = 0
            byhat[LR < 0.5] = 1
            jyhat = np.ones((py[0].shape[0], 1))
            jyhat[LR >= self.model_params[1]] = 0
            jyhat[LR < self.model_params[0]] = 0
            tyhat = 2*jyhat + byhat
            return jyhat, byhat, tyhat
        if self.method == 'FixedSoftmaxThreshold':
            py = self.model.predict(x)
            byhat = np.argmax(py, axis=1)
            jyhat = np.ones((py.shape[0], 1))
            #print(np.max(py, axis=1)> self.model_params)
            jyhat[np.max(py, axis=1) > self.model_params] = 0
            tyhat = self.numClasses*jyhat.reshape(-1) + byhat.reshape(-1)
            return jyhat, byhat, tyhat