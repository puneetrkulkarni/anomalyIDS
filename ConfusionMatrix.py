# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:42:04 2016

@author: pk
@Details: Caculates the Confusion Matrix
"""

def ComputeConfusionMatrix(testResults):
    tp=0
    fp=0
    tn=0
    fn=0
    
    for actual,desired in testResults:
        if actual==desired:
            if actual==0:
                tp=tp+1
            else:
                tn=tn+1
        else:
            if actual==1:
                fp=fp+1                    
            else:
                fn=fn+1 
#==============================================================================
#         if actual==0 and desired==0:
#             tp+=1 #pkts which are expected normal are normal
#         elif actual==1 and desired==0:
#             fp+=1 #pkts which are expected normal are attack
#         elif actual==1 and desired==1:
#             tn+=1 #pkts which are expected attack are attack
#         elif actual==0 and desired==1:
#             fn+=1 #pkts which are expected attack are normal
#==============================================================================
    return tp,fp,tn,fn