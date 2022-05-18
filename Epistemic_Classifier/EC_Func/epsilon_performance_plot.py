from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def epsilon_performance_curve(EC, model, X_test, Y_test, epsilon_list = np.array([0.5, 0.8, 1.2]), plt = plt):
    #metric_cal(y_true, y_pred, raw_pred)
    
    raw_pred = model.predict_classes(X_test)
    y_true = Y_test
    overall_acc = []
    acc_ik = []
    frac_ik = []
    frac_imk = []
    frac_idk = []
    for eps in epsilon_list:
        y_pred = EC.predict_class(X_test, dist=[eps])
        trusted_index = np.where(y_pred < (np.max(y_true)+0.1))[0]
        idk_index = np.where(y_pred == (np.max(y_true).astype(np.int)+1))[0]
        imk_index = np.where(y_pred == (np.max(y_true).astype(np.int)+2))[0]
        # overall accuracy
        overall_acc.append(accuracy_score(y_true, y_pred))
        # fraction of idk
        idk_l = len(idk_index)/len(y_pred)
        imk_l = len(imk_index)/len(y_pred)
        ik_l = len(trusted_index)/len(y_pred)
        
        frac_ik.append(ik_l)
        frac_imk.append(imk_l)
        frac_idk.append(idk_l)
        acc_ik.append(accuracy_score(y_true[trusted_index], y_pred[trusted_index]))
        
    plt.semilogx(epsilon_list, acc_ik, '--', linewidth=2)
    plt.semilogx(epsilon_list, frac_ik, '-', linewidth=2)
    plt.semilogx(epsilon_list, frac_imk, ':', linewidth=2)
    plt.semilogx(epsilon_list, frac_idk, '-.', linewidth=2)
    #plt.plot(epsilon_list, overall_acc)
    plt.legend(['Acc IK', 'Frac IK', 'Frac IMK', 'Frac IDK']) #, 'Overall Acc'
    plt.set_xlabel(r'$\varepsilon$')
    plt.set_ylabel('Fraction')
        
    return overall_acc, acc_ik, frac_ik, frac_imk, frac_idk
    
