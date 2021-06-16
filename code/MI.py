# mutual information

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 07 11:31:10 2018

@author: Xiaoxuan Jia
"""
import numpy as np
from scipy import sparse


#-----------------Mutual information-------\
from sklearn.metrics.cluster import normalized_mutual_info_score

def MI(matrix):
	"""matrix: cell by feature"""
	n==matrix.shape[0]
	MI=np.zeros((n,n))
	for i in range(n):
	    for j in range(n):
	        MI[i,j]=normalized_mutual_info_score(matrix[i,:], matrix[j,:])
	return MI

