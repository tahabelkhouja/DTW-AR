# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:58:45 2020

@author: BkTaha
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import pickle as pkl

from TargetCnn import cnn_class
        
if __name__=="__main__":
    json_param = "UCR_MV_parameters.json"
    json_object = json.load(open(json_param, "r"))
    #Parameters of the dataset
    training_log = open("Exp_MV/training_log.txt", "w")
    for dataset_name in json_object:
    # for dataset_name in ["DodgerLoopDay", "EOGHorizontalSignal", "EthanolLevel", "GestureMidAirD2",
    #                      "GunPointMaleVersusFemale", "Haptics", "InlineSkate", "Mallat",
    #                      "Meat", "OliveOil", "PigAirwayPressure", "PigArtPressure", "Wine"]:
        with open(json_param) as jf:
            info = json.load(jf)
            d = info[dataset_name]
            path = d['path']
            SEG_SIZE = d['SEG_SIZE']
            CHANNEL_NB = d['CHANNEL_NB']
            CLASS_NB = d['CLASS_NB']
            training_log.write("***Dataset: {}: \n".format(dataset_name))
            sys.stdout.write("***Dataset: {}: \n".format(dataset_name))
            sys.stdout.flush()
            #Data Reading
            X_train, y_train, X_test, y_test = pkl.load(open(path, 'rb'))    
            #Model Training
            experim_path = "Exp_MV/Experiment_"+dataset_name
            targetW = cnn_class("WB1", SEG_SIZE, CHANNEL_NB, CLASS_NB, arch='2')
            targetB = cnn_class("BB1", SEG_SIZE, CHANNEL_NB, CLASS_NB, arch='1')
            train_ds = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)).batch(70)
            
            targetW.train(train_ds, checkpoint_path=experim_path+"/TrainingRes/WB_target")
            targetB.train(train_ds, checkpoint_path=experim_path+"/TrainingRes/BB_target")
            res_tn = targetW.score(X_train, y_train)
            res = targetW.score(X_test, y_test)
            res_tnB = targetB.score(X_train, y_train)
            resB = targetB.score(X_test, y_test)
            training_log.write("WB: Training accuracy: {:.3f} - Testing "
                             "accuracy: {:.3f}\n--------".format(res_tn, res))
            training_log.write("BB: Training accuracy: {:.3f} - Testing "
                             "accuracy: {:.3f}\n--------".format(res_tnB, resB))
    training_log.close()
    
    
