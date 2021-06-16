# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 20:18:1 2021

@author: Xiaoxuan Jia
"""

def transfer_entropy(X, Y, lag):
    import numpy as np
    from CPAC.series_mod import cond_entropy
    from CPAC.series_mod import entropy

    # future of i
    Fi = np.roll(X, -lag)
    # past of i
    Pi = X
    # past of j
    Pj = Y

    #Transfer entropy
    Inf_from_Pi_to_Fi = cond_entropy(Fi, Pi)

    #same as cond_entropy(Fi, Pi_Pj)
    Hy = entropy(Pi,Pj)
    Hyx = entropy(Fi,Pj,Pi)
    Inf_from_Pi_Pj_to_Fi = Hyx - Hy

    TE_from_j_to_i = Inf_from_Pi_to_Fi-Inf_from_Pi_Pj_to_Fi

    return TE_from_j_to_i