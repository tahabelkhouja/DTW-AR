import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import pickle as pkl


from absl import app, flags
from TargetCnn import targetModel_cnn

FLAGS = flags.FLAGS

def main(argv):    

    X_train, y_train, X_test, y_test = pkl.load(open("Dataset/"+FLAGS.dataset_name+".pkl", "rb"))
    cnn_model = targetModel_cnn("BaseModel", FLAGS.window_size, FLAGS.channel_dim, FLAGS.class_nb, arch='1')
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(FLAGS.batch_size)
    
    cnn_model.train(train_ds, epochs=FLAGS.epochs, new_train=True,checkpoint_path="TrainedModels/"+cnn_model.name)
    
    res_tn = cnn_model.score(X_train, y_train)
    res = cnn_model.score(X_test, y_test)
    
    
    print("Training accuracy: {:.3f}\nTesting accuracy: {:.3f}".format(res_tn, res))
    
if __name__=="__main__":
    flags.DEFINE_string('dataset_name', None, 'Dataset name')
    flags.DEFINE_integer('window_size', None, 'Window size of the input')
    flags.DEFINE_integer('channel_dim', None, 'Number of channels of the input')
    flags.DEFINE_integer('class_nb', None, 'Total number of classes')
    flags.DEFINE_integer('batch_size', 32, 'Batch Size')
    flags.DEFINE_integer('epochs', 20, 'Epochs number')
    flags.mark_flags_as_required(['dataset_name', 'window_size', 'channel_dim', 'class_nb'])
    app.run(main) 