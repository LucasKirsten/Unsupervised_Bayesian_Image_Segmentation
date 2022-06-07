# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:33:45 2022

@author: kirstenl
"""

import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import mark_boundaries
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from time import time
from itertools import permutations

#%%

# compute discrete gradients of image
def _grad(arr):
    out = np.zeros((2,) + arr.shape, arr.dtype)
    out[0, :-1, :, ...] = arr[1:, :, ...] - arr[:-1, :, ...]
    out[1, :, :-1, ...] = arr[:, 1:, ...] - arr[:, :-1, ...]
    return out

# define total variation
_TV = lambda x:np.sum(np.abs(_grad(x)))

# define method to compute lambda values
_comp_lambda = lambda x:x.size/(_TV(x)+1)

# compute the proposed bayesian segmentation
def bayesian_segmentation(y:np.ndarray, K:int, T:int=50, L:int=25, \
                          eps:float=1e-3, norm:bool=False, verbose:bool=False):
    '''

    Parameters
    ----------
    y : np.ndarray
        Input image.
    K : int
        Number of segments to be detected.
    T : int, optional
        Number of outer iterations. The default is 50.
    L : int, optional
        Number of inner iterations. The default is 25.
    eps : float, optional
        tolerance. The default is 1e-3.
    norm : bool, optional
        If to normalize lambda value to compute chambolle algorithm in very noisy images. The default is False.
    verbose : bool, optional
        If to display the progress. The default is False.

    Returns
    -------
    newz : np.ndarray
        Final segmentation.

    '''
    
    init = time()
    
    # initialize mean and z values
    z = np.zeros_like(y)
    u = np.zeros_like(y)
    
    # define initial condition
    x = 2*y
    
    # outer loop
    for t in range(1,T):
        if verbose: print(t, end='...')
        
        lambdax = _comp_lambda(x)
        v = np.copy(x)
        
        # inner iterations
        for l in range(L):
            if verbose: print(l, end='.')
            
            # compute chambolle algorithm
            lambdal = _comp_lambda(v)
            v = denoise_tv_chambolle(v+u, weight=1/lambdal if norm else lambdal)
            
            # verify condition of inner loop
            if _comp_lambda(v) - lambdax < eps*lambdax:
                break
        if verbose: print()
        
        x = np.copy(v)
        
        # compute the clustering for the mean and z vectors
        kmeans = KMeans(n_clusters=K)
        pred = kmeans.fit_predict(x.reshape(-1,1))
        cluster = np.squeeze(kmeans.cluster_centers_)
        
        # adjust values to the predictions
        u = cluster[pred].reshape(y.shape)
        newz = pred.reshape(y.shape)
        
        # verify the similarity between z_before and z_now value
        sim = np.count_nonzero(newz==z)/z.size
        if sim>0.99:
            break
        else:
            z = newz
        
        if verbose: print('Elapsed time: ', time()-init)
        
        return newz
            

#%%

if __name__=='__main__':
    
    # define the datasets (name, if to norm, K, if to mark plot)
    datasets = [
        ('GMM4', True, 4, False),
        ('GMM8', True, 8, False),
        ('LMM2', True, 2, False),
        ('PMM3', True, 3, False),
        ('Bacteria', False, 2, True),
        ('Brain', False, 3, True),
        ('Lungs', False, 2, True),
        ('SAR', True, 3, True)
    ]
    
    # define colors for marks in segmentations
    colors = [(255,0,0), (0,255,0)]
    
    # evaluate on all datasets
    for data,norm,K,marks in datasets:
        print(f'Evaluating {data}...')
        
        path_img = f'data/{data}.png'
        path_gt  = f'data/{data}_gt.npy'
        
        # verify for ground truth
        has_gt = os.path.exists(path_gt)
    
        # load data and ground truth (if any)
        y = np.float64(cv2.imread(path_img, cv2.IMREAD_GRAYSCALE))
        if has_gt:
            gt = np.load(path_gt).astype('int32')
            y = cv2.resize(y, gt.shape[::-1])
        
        # apply bayesian segmentation
        segs = bayesian_segmentation(y=y, K=K, norm=norm)
        
        # display marked segmentations
        if has_gt and marks:
            plt.figure()
            marks = mark_boundaries(np.copy(y), gt, color=(255,0,0)).astype('uint8')
            plt.imshow(marks)
            plt.title('Ground truth')
        
        if marks:
            marks = np.copy(y)
            for i,c in enumerate(np.unique(segs)[1:]):
                inp_seg = np.where(segs==c, 1, 0)
                outline_color = colors[i]
                marks = mark_boundaries(marks, inp_seg, color=colors[i], outline_color=outline_color)
                
            plt.figure()
            plt.subplot(121)
            plt.imshow(y); plt.title(data)
            
            plt.subplot(122)
            plt.imshow(marks.astype('uint8')); plt.title('Predicted')
            plt.show()
        
        
        # evaluate result (it can take a while to compute automatically depending on K)
        if has_gt:
            print('Evaluating results...')
            pred = np.copy(segs)
            final_pred = np.copy(pred)
            acc = 0
            for pred_val in permutations(np.arange(0,K), K):
                
                pp = np.copy(pred)
                for i,val in enumerate(pred_val):
                    pp = np.where(pred==i, val, pp)
                
                nowacc = np.count_nonzero(gt==pp)*100/gt.size
                if nowacc>acc:
                    acc = nowacc
                    final_pred = np.copy(pp)
        
            # display image of final segmentations
            plt.figure()
            plt.subplot(131)
            plt.imshow(y); plt.title(data)
            
            plt.subplot(132)
            plt.imshow(gt); plt.title('Ground truth')
            
            plt.subplot(133)
            plt.imshow(final_pred); plt.title('Predicted')
            plt.show()
        
            print('Accuracy= ', acc)
            print()










