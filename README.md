# Dynamic Time Warping based Adversarial Framework for Time-Series Domain
Python Implementation of Dynamic Time Warping for Adversarial Robustness (DTW-AR) for the paper: "[Dynamic Time Warping for Adversarial Robustness (DTW-AR)Adversarial Framework with Certified Robustness for Time-Series Domain via Statistical Features]()" by Taha Belkhouja, Yan Yan and Janardhan Rao Doppa.

## Setup 
```
pip install -r requirement.txt
```
By default, data is stored in `experim_path_{dataset_name}`. Directory can be changed in `RO_TS.py`


## Obtain datasets
- The dataset can be obtained as .zip file from "[The UCR Time Series Classification Repository](http://www.timeseriesclassification.com/dataset.php)".
- Download the .zip file and extract it it in `Dataset/{dataset_name}` directory.
- The data should be stored in `Dataset` directory in [pickle](https://docs.python.org/3/library/pickle.html) format containing the training-testing examples with their corresponding labels as follow: X_train, y_train, X_test, y_test

## Run
- Example DTW-AR adversarial attack run
```
python run_dtwar.py --dataset_name ERing --window_size 65 --channel_dim 4 --class_nb 6 --dtw_window 3 --iter 2500
```
The adversarial attack will be stored in [pickle](https://docs.python.org/3/library/pickle.html) file "DTWAR_Attack.pkl"