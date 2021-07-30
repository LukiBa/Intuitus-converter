# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:11:16 2019

@author: Lukas, Based on: Tutorial: Tutorial : Facial Expression Classification Keras from Kaggle
"""
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import brewer2mpl
from sklearn.metrics import confusion_matrix
import os


def overview(start, end, X):
    """
    The function is used to plot first several pictures for overviewing inputs format
    """
    fig = plt.figure(figsize=(20,20))
    for i in range(start, end+1):
        input_img = X[i:(i+1),:,:,:]
        ax = fig.add_subplot(16,12,i+1)
        ax.imshow(input_img[0,:,:,0], cmap=plt.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    plt.show()
    
def show_single(X,n):
    """
    Plots single image of 4-dimensional imgae dataset.

    Parameters
    ----------
    X : 4-dim numpy array 
        Dataset.
    n : integer
        index of image.

    Returns
    -------
    None.

    """
    input_img = X[n:n+1,:,:,:] 
    print ('Image shape: {}'.format(input_img.shape))
    plt.imshow(input_img[0,:,:,0], cmap='gray')
    plt.show()

def plot_evaluation(epochs,acc,val_acc,loss,val_loss,MODEL_NAME):
    """
    Plots the evaluation data of a NN

    Parameters
    ----------
    epochs : Iterator
        An iterator over the epochs.
    acc : numpy array
        accuarcy of the epochs.
    loss : numpy array
        loss of the epochs.
    val_loss : numpy array
        val_loss of the epochs

    Returns
    -------
    None.

    """
    plt.plot(epochs, acc, 'bo', label='Training acc')    
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()    
    plt.savefig('{}/evalutate_accuracy.png'.format(MODEL_NAME))
    plt.figure()   
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()    
    plt.savefig('{}/evalutate_loss.png'.format(MODEL_NAME))  
    plt.show()  
      
    
def plot_distribution(y1, y2, data_names,labels, MODEL_NAME,ylims =[1000,1000]): 
    """
    The function is used to plot the distribution of the labels of provided dataset 
    """
    colorset = brewer2mpl.get_map('Set3', 'qualitative', 6).mpl_colors
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(1,2,1)
    ax1.bar(np.arange(1,7), np.bincount(y1), color=colorset, alpha=0.8)
    ax1.set_xticks(np.arange(1.25,7.25,1))
    ax1.set_xticklabels(labels, rotation=60, fontsize=14)
    ax1.set_xlim([0, 8])
    ax1.set_ylim([0, ylims[0]])
    ax1.set_title(data_names[0])
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.bar(np.arange(1,7), np.bincount(y2), color=colorset, alpha=0.8)
    ax2.set_xticks(np.arange(1.25,7.24,1))
    ax2.set_xticklabels(labels, rotation=60, fontsize=14)
    ax2.set_xlim([0, 8])
    ax2.set_ylim([0, ylims[1]])
    ax2.set_title(data_names[1])
    plt.tight_layout()
    plt.savefig('{}/distribution.png'.format(MODEL_NAME)) 
    plt.show()

def plot_subjects(X,start, end, y_pred, y_true,fig, title=True):
    """
    The function is used to plot the picture subjects
    """
    #fig = plt.figure(figsize=(12,12))
    #emotion = {0:'Angry', 1:'Disgust' , 2:'Fear' , 3:'Happy', 4:'Sad', 
    #               5:'Surprise', 6:'Neutral'}
    emotion = {0:'Angry', 1:'Fear', 2:'Happy', 3:'Sad', 4:'Surprise', 5:'Neutral'}
    for i in range(start, end+1):
        input_img = X[i:(i+1),:,:,:]
        ax = fig.add_subplot(6,6,i+1)
        ax.imshow(input_img[0,:,:,0], cmap=matplotlib.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        if y_pred[i] != y_true[i]:
            plt.xlabel(emotion[y_true[i]], fontsize=12)
        else:
            plt.xlabel(emotion[y_true[i]], fontsize=12)
        if title:
            if y_pred[i] != y_true[i]:
                plt.title(emotion[y_pred[i]], color='red', fontsize=12)
            else:
                plt.title(emotion[y_pred[i]], color='green', fontsize=12)   
        plt.tight_layout()
    #plt.savefig('{}/subjects'.format(MODEL_NAME)+'{}.png'.format(fig_num))     
    #plt.show() 
    
def plot_probs(X,start,end, y_prob,labels,fig):
    """
    The function is used to plot the probability in histogram for six labels 
    """
    #fig = plt.figure(figsize=(12,12))
    for i in range(start, end+1):
        input_img = X[i:(i+1),:,:,:]
        ax = fig.add_subplot(6,6,i+1)
        set3 = brewer2mpl.get_map('Set3', 'qualitative', 6).mpl_colors
        ax.bar(np.arange(0,6), y_prob[i], color=set3,alpha=0.5)
        ax.set_xticks(np.arange(0.5,6.5,1))
        #labels = ['angry', 'fear', 'happy', 'sad', 'surprise','neutral']
        ax.set_xticklabels(labels, rotation=90, fontsize=10)
        ax.set_yticks(np.arange(0.0,1.1,0.5))
        plt.tight_layout()
    #plt.savefig('{}/probes'.format(MODEL_NAME)+'{}.png'.format(fig_num))    
    #plt.show()    
    
def plot_subjects_with_probs(X,start, end, y_prob, y_pred, y_true,labels,MODEL_NAME):
    """
    This plotting function is used to plot the probability together with its picture
    """
    iter = int((end - start)/6)
    fig_sub = plt.figure(figsize=(12,12))
    for i in np.arange(0,iter):
        plot_subjects(X,i*6,(i+1)*6-1, y_pred, y_true,fig_sub, title=True)
    fig_sub.savefig('{}/Subjects.png'.format(MODEL_NAME)) 
    
    fig_prob = plt.figure(figsize=(12,12))
    for i in np.arange(0,iter):    
        plot_probs(X,i*6,(i+1)*6-1, y_prob,labels,fig_prob)    
    fig_prob.savefig('{}/Probabilty.png'.format(MODEL_NAME)) 
    
def plot_distribution2(y_true, y_pred,labels,MODEL_NAME):
    """
    The function is used to compare the number of true labels as well as prediction results
    """
    colorset = brewer2mpl.get_map('Set3', 'qualitative', 6).mpl_colors
    ind = np.arange(1.5,7,1)  # the x locations for the groups
    width = 0.35   
    fig, ax = plt.subplots()
    true = ax.bar(ind, np.bincount(y_true), width, color=colorset, alpha=1.0)
    pred = ax.bar(ind + width, np.bincount(y_pred), width, color=colorset, alpha=0.3)
    ax.set_xticks(np.arange(1.5,7,1))
    ax.set_xticklabels(labels, rotation=30, fontsize=14)
    ax.set_xlim([1.25, 7.5])
    ax.set_ylim([0, 1000])
    ax.set_title('True and Predicted Label Count (Private)')
    plt.tight_layout()
    plt.savefig('{}/distribution_compare.png'.format(MODEL_NAME)) 
    plt.show()        
    
def plot_confusion_matrix(y_true, y_pred, labels,MODEL_NAME, cmap=plt.cm.Blues):
    """
    The function is used to construct the confusion matrix 
    """
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6,6))
    matplotlib.rcParams.update({'font.size': 16})
    ax  = fig.add_subplot(111)
    matrix = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(matrix) 
    for i in range(0,6):
        for j in range(0,6):  
            ax.text(j,i,cm[i,j],va='center', ha='center')
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('{}/Confusion_matrix.png'.format(MODEL_NAME)) 
    plt.show()

    