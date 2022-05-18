import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import pandas as pd

def metric_cal(y_true, y_pred, raw_pred):
    
    trusted_index = np.where(y_pred < (np.max(y_true)+0.1))[0]
    idk_index = np.where(y_pred == (np.max(y_true).astype(np.int)+1))[0]
    imk_index = np.where(y_pred == (np.max(y_true).astype(np.int)+2))[0]
    
    cm1 = confusion_matrix(np.concatenate([y_true[trusted_index], list(range(np.max(y_true)+1))]), 
                           np.concatenate([y_pred[trusted_index], list(range(np.max(y_true)+1))])).T
    
    cm2 = confusion_matrix(np.concatenate([y_true[imk_index], list(range(np.max(y_true)+1))]), 
                           np.concatenate([raw_pred[imk_index], list(range(np.max(y_true)+1))])).T
    
    cm3 = confusion_matrix(np.concatenate([y_true[idk_index], list(range(np.max(y_true)+1))]), 
                           np.concatenate([raw_pred[idk_index], list(range(np.max(y_true)+1))])).T
    
    cm1 = cm1 - np.identity(cm1.shape[0]).astype(np.int)
    cm2 = cm2 - np.identity(cm1.shape[0]).astype(np.int)
    cm3 = cm3 - np.identity(cm1.shape[0]).astype(np.int)


    M1 = len(trusted_index)/len(y_true)
    if np.sum(cm1) == 0:
        M2 = np.sum(np.diag(cm1))/(np.sum(cm1)+1e-9)
    else:
        M2 = np.sum(np.diag(cm1))/(np.sum(cm1))
       
    if np.sum(cm2) == 0:
        M3 = np.sum(np.diag(cm2))/(np.sum(cm2)+1e-9)
    else:
        M3 = np.sum(np.diag(cm2))/(np.sum(cm2))
        
    M4 = len(imk_index)/len(y_true)
    M5 = len(idk_index)/len(y_true)
    
    if np.sum(cm2+cm3) == 0:
        M6 = np.sum(np.diag(cm2+cm3))/(np.sum(cm2+cm3)+1e-9)
    else:
        M6 = np.sum(np.diag(cm2+cm3))/(np.sum(cm2+cm3))
    
    print('M1: ', M1)
    print('M2: ', M2)
    print('M3: ', M3)
    print('M4: ', M4)
    print('M5: ', M5)
    print('M6: ', M6)
    
    return M1, M2, M3, M4, M5, M6
    

def plot_confusion_matrix(y_true, y_pred, raw_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          ep=0):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
#     M1, M2, M3, M4, M5, M6 = metric_cal(y_true, y_pred, raw_pred)

    # Compute confusion matrix
    
    trusted_index = np.where(y_pred < (np.max(y_true)+0.1))[0]
    idk_index = np.where(y_pred == (np.max(y_true).astype(np.int)+1))[0]
    imk_index = np.where(y_pred == (np.max(y_true).astype(np.int)+2))[0]
    
    cm1 = confusion_matrix(np.concatenate([y_true[trusted_index], list(range(np.max(y_true)+1))]), 
                           np.concatenate([y_pred[trusted_index], list(range(np.max(y_true)+1))])).T
    
    cm2 = confusion_matrix(np.concatenate([y_true[imk_index], list(range(np.max(y_true)+1))]), 
                           np.concatenate([raw_pred[imk_index], list(range(np.max(y_true)+1))])).T
    
    cm3 = confusion_matrix(np.concatenate([y_true[idk_index], list(range(np.max(y_true)+1))]), 
                           np.concatenate([raw_pred[idk_index], list(range(np.max(y_true)+1))])).T
    
    
    cm1 = cm1 - np.identity(cm1.shape[0]).astype(np.int)
    cm2 = cm2 - np.identity(cm1.shape[0]).astype(np.int)
    cm3 = cm3 - np.identity(cm1.shape[0]).astype(np.int)
    
    cm = np.concatenate([cm1, cm2, cm3])
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels= np.concatenate([list(range(np.max(y_true)+1)), list(range(np.max(y_true)+1)), list(range(np.max(y_true)+1))]),
           yticklabels=np.concatenate([list(range(np.max(y_true)+1)), list(range(np.max(y_true)+1)), list(range(np.max(y_true)+1))]),
           ylabel='Predicted label',
           xlabel='True label',
           ylim = (-0.5, cm.shape[0]-0.5))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.invert_yaxis()
    plt.show()
    fig.savefig(fname = 'acm_'+str(ep)+'.png')    
    #     return M1, M2, M3, M4, M5, M6


def plot_confusion_matrix_baselines(y_true, y_pred, raw_pred = None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    #metric_cal(y_true, y_pred, raw_pred)
    if len(y_pred.shape)>1:
        y_pred = y_pred[:, 0].astype(np.int)
    else:
        y_pred = y_pred.astype(np.int)
        
    index_unsure = np.where(y_pred>np.max(y_true))[0]
    index_sure = np.where(y_pred<np.max(y_true)+0.1)[0]
    
    y_pred[index_unsure] = y_pred[index_unsure] - np.max(y_true) - 1
    y_pred = y_pred.astype(np.int)
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    
    trusted_index = index_sure
    idk_index = index_unsure
    
    
    cm1 = confusion_matrix(np.concatenate([y_true[trusted_index], list(range(np.max(y_true)+1))]), 
                           np.concatenate([y_pred[trusted_index], list(range(np.max(y_true)+1))])).T
    
    cm2 = confusion_matrix(np.concatenate([y_true[idk_index], list(range(np.max(y_true)+1))]), 
                           np.concatenate([y_pred[idk_index], list(range(np.max(y_true)+1))])).T
    
    
    cm1 = cm1 - np.identity(cm1.shape[0]).astype(np.int)
    cm2 = cm2 - np.identity(cm1.shape[0]).astype(np.int)
    
    cm = np.concatenate([cm1, cm2])
    
    M1 = len(trusted_index)/len(y_true)
    if np.sum(cm1) == 0:
        M2 = np.sum(np.diag(cm1))/(np.sum(cm1)+1e-9)
    else:
        M2 = np.sum(np.diag(cm1))/(np.sum(cm1))
       
    if np.sum(cm2) == 0:
        M3 = np.sum(np.diag(cm2))/(np.sum(cm2)+1e-9)
    else:
        M3 = np.sum(np.diag(cm2))/(np.sum(cm2))
    
    
    M4 = 0
    M5 = 0
    M6 = M3
    
    print('M1: ', M1)
    print('M2: ', M2)
    print('M3: ', M3)
    
    
    fig, ax = plt.subplots(figsize=(18, 9))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels= np.concatenate([list(range(np.max(y_true)+1)), list(range(np.max(y_true)+1))]), 
           yticklabels=np.concatenate([list(range(np.max(y_true)+1)), list(range(np.max(y_true)+1))]),
           title=title,
           ylabel='Predicted label',
           xlabel='True label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    plt.show()
    
    return M1, M2, M3, M4, M5, M6
