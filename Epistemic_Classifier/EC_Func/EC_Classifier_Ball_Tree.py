from keras import backend as K
from sklearn.neighbors import BallTree
import numpy as np
from numpy import linalg as LA
from multiprocessing import Pool

# ''''''''''''''''''''''''''''''''''''''''
def querry_parallel_knn(tree_data):
    tree = tree_data[0]
    hidden = tree_data[1]
    para = tree_data[2]
    mode = tree_data[3]
    if mode == 'knn':
        dist, knn = tree.query(hidden, k=para)
    else:
        knn = tree.query_radius(hidden, r=para)

    return knn


def querry_parallel_nn(tree, hidden, knn, process_num, mode='knn'):
    split_data = np.array_split(hidden, process_num)

    tuple_model = [(tree, spl_x, knn, mode) for spl_x in split_data]

    proc = Pool(process_num)
    res_list = proc.map(querry_parallel_knn, tuple_model)

    return np.concatenate(res_list, axis=0)


# ''''''''''''''''''''''''''''''''''''''''


def get_activation(img, model, layer_select, verbose=1):
    '''
    extract activation from the outputs of selected layer given input
    :param img:
    :param model:
    :param layer_select:
    :param verbose: whether output name of the selected layer
    :return:
    '''

    inp = model.input  # input placeholder
    outputs = []
    if verbose == 1:
        print('layer selected: ')
    for i in layer_select:
        outputs.append(model.layers[i].output)
        if verbose == 1:
            print(model.layers[i].name)

    eval_function = K.function([inp, K.learning_phase()], outputs)  # evaluation function
    result = eval_function([img, 0.])
    data_all = []
    for activation in result:
        # if extracted layer is convolutional layer
        if len(activation.shape) == 4:
            res = np.reshape(activation, (img.shape[0], -1))
        # print(np.reshape(layer_output, (x.shape[0], -1)))
        else:
            res = activation

        # print(res.shape)
        data_all.append(res)

    return data_all


def ellipse_metric(W):
    '''
    Distance matrix
    :param W:
    :return:
    '''

    w_v, v_v = LA.eig(W.dot(W.T))
    eig = w_v[np.abs(w_v) > 1e-10]
    eig_vector = v_v[:, np.abs(w_v) > 1e-10]
    trans_eig = eig_vector.T.dot(W).real
    tmp = np.zeros(trans_eig.shape)
    for index, vec_eig in enumerate(trans_eig):
        tmp[index] = vec_eig / np.sqrt(vec_eig.dot(vec_eig.T))

    v_low = tmp.T
    distance_metric = v_low.dot(np.diag(1 / eig)).dot(v_low.T)
    return distance_metric.real


class EpistemicClassifier:
    '''
    V2 of Epistemic classifier
    '''

    def __init__(self, model, layer_select, metric='minkowski', p=2, process_num=1):
        self.model = model
        self.n_dknn_layers = len(layer_select)
        self.selected_layer = layer_select
        self.tree_list = None
        self.label_list = None
        self.metric = metric
        self.process_num = process_num
        self.p = p  # p norm

    def fit(self, train_x, train_y):
        '''
        Epistemic model fitting given training and testing data
        :param train_x:
        :param train_y:
        :return:
        '''
        print('Using metric:', self.metric)
        print('caching hiddens')
        result_act = get_activation(train_x, self.model, self.selected_layer)
        self.act_list = result_act
        self.label_list = train_y

        if self.metric == 'mahalanobis':
            invCov = []
            W = np.identity(self.model.layers[0].get_weights()[0].shape[0])
            for layer_index in range(np.max(self.selected_layer) + 1):
                W = W.dot(self.model.layers[layer_index].get_weights()[0])
                if layer_index in self.selected_layer:
                    invCov.append(ellipse_metric(W))

        print('using Ball Tree for NN Search')
        self.tree_list = []  # one lookup tree for each layer
        for i in range(self.n_dknn_layers):
            print('building tree for layer {}'.format(i))
            if self.metric == 'minkowski':
                tree = BallTree(self.act_list[i], metric=self.metric, p=self.p)

            elif self.metric == 'mahalanobis':  # using Mahalanobis for short-cut of metric matrix
                tree = BallTree(self.act_list[i], metric=self.metric, VI=invCov[i])

            else:
                print("Not yet implemented.............use L2 distance", self.metric)
                tree = BallTree(self.act_list[i])

            self.tree_list.append(tree)

    ''' returns the indices of the nearest neighbors according
    to their position in the training data'''

    def get_neighbors(self, xs, n_neigh=None, dist_v=None):
        '''
        Input data and return back label index of the data
        :param xs:
        :param n_neigh:
        :param dist_v:
        :return:
        '''
        assert self.tree_list is not None
        assert self.label_list is not None

        EC_layers = get_activation(xs, self.model, self.selected_layer, verbose=0)
        neighbors = []
        for layer_id, hidden in enumerate(EC_layers):
            # go through layers and get neighbors for each

            if self.process_num <= 1:
                if dist_v == None:
                    dist, knn = self.tree_list[layer_id].query(hidden, k=n_neigh[layer_id])
                    knn = [np.array(ar) for ar in knn]
                    neighbors.append(knn)

                else:
                    knn = self.tree_list[layer_id].query_radius(hidden, r=dist_v[layer_id])
                    knn = [np.array(ar) for ar in knn]
                    neighbors.append(knn)
            else:
                # querry_parallel_nn(tree, hidden, knn, process_num, mode='knn')
                if dist_v == None:
                    knn = querry_parallel_nn(self.tree_list[layer_id], hidden, n_neigh[layer_id], self.process_num, mode='knn')
                    knn = [np.array(ar) for ar in knn]
                    neighbors.append(knn)

                else:
                    knn = querry_parallel_nn(self.tree_list[layer_id], hidden, dist_v[layer_id], self.process_num,mode='dist')
                    knn = [np.array(ar) for ar in knn]
                    neighbors.append(knn)

        return neighbors

    def predict_neigh(self, xs, n_neigh=None):
        '''
        Prediction based on k nearest neighbor
        :param xs:
        :param n_neigh:
        :return:
        '''

        imk_c = np.max(self.label_list).astype(np.int) + 2
        neigh_list = self.get_neighbors(xs, n_neigh=n_neigh)
        neigh_list = np.array(neigh_list)
        pred_label = np.argmax(self.model.predict(xs), axis=1)
        data_point = neigh_list.shape[1]
        tmp_label = np.ones(data_point).astype(np.int) * imk_c
        tmp_label = tmp_label.astype(np.int)
        for i in range(len(tmp_label)):
            epi_label = self.label_list[np.concatenate(neigh_list[:, i])]
            set_label = list(set(epi_label))
            if (len(set_label) == 1) and (set_label[0] == pred_label[i]):
                tmp_label[i] = pred_label[i]

            # if prediction is not in the set of justification, then it is IDK
            if (pred_label[i] in set_label) == False:
                tmp_label[i] = imk_c - 1

        pred_label = tmp_label
        return pred_label.astype(np.int)

    def predict_dist(self, xs, dist=None):
        '''
        Prediction based on the neighbor in distance
        :param xs:
        :param dist:
        :return:
        '''
        imk_c = np.max(self.label_list).astype(np.int) + 2
        neigh_list = self.get_neighbors(xs, dist_v=dist)
        #print(neigh_list, len(neigh_list))

        tmp = np.array(neigh_list)
        if len(tmp.shape) == 3:
            print("use beta testing function")
            neigh_list.append([[np.array([1]) for i in range(tmp.shape[2] + 1)] for j in range(xs.shape[0])])
            neigh_list = np.array(neigh_list)
            neigh_list = neigh_list[:-1, :]
            print(neigh_list.shape)

        neigh_list = np.array(neigh_list)

        assert len(neigh_list.shape) == 2

        pred_label = np.argmax(self.model.predict(xs), axis=1)
        data_point = neigh_list.shape[1]
        tmp_label = np.ones(data_point).astype(np.int) * imk_c
        tmp_label = tmp_label.astype(np.int)

        for i in range(len(tmp_label)):
            # if one of layer has a empty set, prediction is IDK
            label_tmp = imk_c
            for neig in neigh_list[:, i]:
                if len(neig) == 0:
                    label_tmp = imk_c - 1

            if label_tmp == imk_c:
                # if all layer has agreement on justification, which also agree with prediction
                epi_label = self.label_list[np.concatenate(neigh_list[:, i])]
                set_label = list(set(epi_label))
                if (len(set_label) == 1) and (set_label[0] == pred_label[i]):
                    label_tmp = pred_label[i]

                # if layer is not pure, but intersection of justification and prediction is empty, then it is IDK
                if (pred_label[i] in set_label) == False:
                    label_tmp = imk_c - 1

            tmp_label[i] = label_tmp

        pred_label = tmp_label
        return pred_label.astype(np.int)

    def predict_class(self, xs, n_neigh=None, dist=[], adaptive_ball=False,
                      discount_radius=None, epsilon=0):

        if discount_radius == None:
            discount_radius = [1 for _ in range(len(self.selected_layer))]

        if adaptive_ball:
            # pred epsilon label layer-wise, np.sqrt(lambda)*epsilon (lambda is eigen value of w *  w.T)
            dist = []
            W = np.identity(self.model.layers[0].get_weights()[0].shape[0])
            discount_index = 0
            for layer_index in range(np.max(self.selected_layer) + 1):
                W = W.dot(self.model.layers[layer_index].get_weights()[0])
                if layer_index in self.selected_layer:
                    eig_v, eig_vect = LA.eig(W.dot(W.T))
                    dist.append(np.abs(np.sqrt(np.max(eig_v))) * epsilon * discount_radius[discount_index])
                    discount_index += 1

            print('adaptive distance in each layer', dist)
            return self.predict_dist(xs, dist=dist)

        if self.metric == 'mahalanobis' and n_neigh == None:
            print('adaptive ellipse dist - under testing', dist)
            return self.predict_dist(xs, dist=dist)

        if n_neigh != None:
            if isinstance(n_neigh, list) == False:
                tmp = [n_neigh for _ in range(len(self.selected_layer))]
                n_neigh = tmp

            return self.predict_neigh(xs, n_neigh=n_neigh)

        if len(dist) > 1:
            return self.predict_dist(xs, dist=dist)
        elif len(dist) == 1:
            return self.predict_dist(xs, dist=[dist]*len(self.selected_layer))
            


    def hybrid_predict(self, xs, n_neigh=None, dist=None, mode='IK_intersection'):
        '''
        Hybrid support prediction
        :param xs:
        :param n_neigh:
        :param dist:
        :param mode:
        :return:
        '''
        pred_func = self.predict_class
        pred1 = None
        imk_c = np.max(self.label_list).astype(np.int) + 2
        idk_c = imk_c - 1
        if mode == 'IK_intersection':
            pred1 = pred_func(xs, dist=dist)  # list represent epi in each layer
            pred2 = pred_func(xs, n_neigh=n_neigh)
            pred1[np.where(pred1 != pred2)[0]] = imk_c
        elif mode == 'IDK_union':
            pred1 = pred_func(xs, dist=dist)  # list represent epi in each layer
            pred2 = pred_func(xs[np.where(pred1 == idk_c)[0]], n_neigh=n_neigh)
            pred1[np.where(pred1 == idk_c)[0]] = pred2
        else:
            print("not implemented")

        return pred1