# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 21:54:26 2016
Static Config Parameters
@author: pk
"""

""" Network Definition """
no_of_layers=4 #Including Input and OutPut layer
no_of_inputs=38 #Define Number of Input Layers
no_of_outputs=1 #Define No.final Outputs required ; 
processing_units_hiddenlayer_1=2
processing_units_hiddenlayer_2=4
processing_units_hiddenlayer_3=5
processing_units_hiddenlayer_4=3
#processing_units.hiddenlayer2=1
#processing_units.hiddenlayer3=1

#filename='30000_records_file.data'
#eta=1.0
#epoches=10
#batch_size=4

""" for 10000 record datasetr file """
#filename='10000_records_file.data'
#eta=1.0
#epoches=10
#batch_size=4

""" for 2 L record Dataset file """
filename='10000_records_file.data'
eta=3.0
epoches=1
batch_size=100

""" Speific to Dataset """
col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
                                     "logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
                                     "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
                                     "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
                                     "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                                     "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
                                     
numerical_features = ["duration","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
                 "logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
                 "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
                 "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
                 "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                 "dst_host_rerror_rate","dst_host_srv_rerror_rate"]