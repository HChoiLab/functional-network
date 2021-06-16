# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 20:20:10 2021

@author: Xiaoxuan Jia
"""

from statsmodels.tsa.stattools import grangercausalitytests
# calculate granger causality in time domain
#x : array, 2d, (nobs,2)
#data for test whether the time series in the second column Granger causes the time series in the first column
#results : dictionary, keys arze the number of lags. 
#ssr-based F test is the "standard" granger causality test
maxlag=100
n=np.shape(matrix)[0]
GC=np.zeros((n,n,maxlag))
for i in range(n):
    for j in range(n):
        # bidirection interaction, but no auto
        if i!=j:
            # mean across orientation and repeats
            x=matrix[[i,j],:,:,20:].mean(1).mean(1).T
            G = grangercausalitytests(x, maxlag=maxlag, addconst=True, verbose=False)
            # index in list comprehension is also global, need to be careful
            fscore = [G[g][0]['ssr_ftest'][0] for g in np.arange(1,len(G)+1)]
            GC[i,j,:]=fscore
