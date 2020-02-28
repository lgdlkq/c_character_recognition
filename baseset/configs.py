'''
@env_version: python3.5.2
@Author: 雷国栋
@LastEditors: 雷国栋
@Date: 2020-02-10 16:51:14
@LastEditTime: 2020-02-22 15:59:14
'''
import os

base_side = 'G:/PythonTrainFaile/c_character_recognition/'
train_side = base_side + 'train/'

model_name = 'ResNetX'
root = os.getcwd()
weights = root + '/checkpoints/'
best_models = weights + '/best/'
logs = root + '/logs/'
tf_logs = logs + 'tf_logs/'

gpus = '0'

train_ratio = 0.85
valid_ratio = 0.075
test_ratio = 0.075

img_size = (224, 224)
batch_size = 12
epochs = 50

shuffle = True
num_workers = 0
seed = 888
lr = 1e-4
final_lr = 1e-3
weight_decay = 0.0003
