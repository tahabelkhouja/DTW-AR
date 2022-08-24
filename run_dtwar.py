import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import json
import argparse
import copy

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import pickle as pkl

from tqdm import tqdm
from TargetCnn import cnn_class

def attack(args):
    json_param = "datasets_parameters.json"
    dataset_name = args.dataset_name
    with open(json_param) as jf:
        info = json.load(jf)
    info_dict = info[dataset_name]
    path = info_dict['path']
    SEG_SIZE = info_dict['SEG_SIZE']
    CHANNEL_NB = info_dict['CHANNEL_NB']
    CLASS_NB = info_dict['CLASS_NB']

    print(f"DTW-AR attack algorithm on {args.dataset_name}".format(dataset_name))
    X_train, y_train, X_test, y_test = pkl.load(open(path, 'rb'))
    experim_path = os.path.join("Experiments", "Experiment_"+dataset_name)
    if not os.path.isdir(experim_path):
        os.makedirs(experim_path)
        os.makedirs(os.path.join(experim_path, "TrainingRes"))

    target = cnn_class("WB", SEG_SIZE, CHANNEL_NB, CLASS_NB, arch='2')
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(args.epochs)
    existing_models = [mdl for mdl in os.listdir(os.path.join(experim_path, "TrainingRes"))]
    if any([mdl_name.startswith(args.model_name) for mdl_name in existing_models]):
        target.train([], new_train=False, checkpoint_path=os.path.join(experim_path, "TrainingRes", args.model_name))
        print(f"{args.model_name} weights are loaded!")
    else:
        target.train(train_ds, new_train=True, epochs=args.epochs, checkpoint_path=os.path.join(experim_path, "TrainingRes", args.model_name))
    acc = target.score(X_test, y_test)
    print(f"Current model accuracy {acc:.2}")

    attack_file = os.path.join(experim_path, "DTWAR_attack.pkl")
    if os.path.isfile(attack_file):
        X_ind, X_natural, X_adversarial, y_natural, y_adversarial = pkl.load(open(attack_file, 'rb'))
        print("Resuming")
    else: 
        X_ind, X_natural, X_adversarial, y_natural, y_adversarial = 0, np.ndarray((0,)+X_train.shape[1:]), np.ndarray((0,)+X_train.shape[1:]), np.array([]), np.array([])

    
    for sample_id, X in enumerate(tqdm(X_train, desc="DTW-AR Run")):
        if sample_id < X_ind:
            continue
        X_nat = copy.deepcopy(X[np.newaxis, :])
        X = tf.convert_to_tensor(X_nat)
        y_nat = y_train[sample_id]

        y_adv = np.random.randint(0, CLASS_NB)
        while y_nat==y_adv:
            y_adv = np.random.randint(0, CLASS_NB)
        X_adv = target.dtwar_attack(X, y_adv, rho=-5, alpha=0.05, beta=0.05, eta=5e-1, delta_l2_loss=1, max_iter=args.iter)

        X_natural = np.concatenate([X_natural, X_nat])
        X_adversarial = np.concatenate([X_adversarial, X_adv])
        y_natural = np.concatenate([y_natural, [y_nat]])
        y_adversarial = np.concatenate([y_adversarial, [y_adv]])
        pkl.dump([sample_id, X_natural, X_adversarial, y_natural, y_adversarial], open(attack_file, "wb"))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help="Dataset name", required=True)
    parser.add_argument('--model_name', type=str, default="WB", help="DNN model name", required=False)
    parser.add_argument('--epochs', type=int, default=70, help="Number of training epochs", required=False)
    ## DTW-AR hyper-parameters
    parser.add_argument('--iter', type=int, default=1000, help="Number of maximum iterations per attack", required=False)
    args = parser.parse_args()
    attack(args)