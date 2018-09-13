import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


dataset_dir = '/Users/ahmedtaha/Documents/dataset/'
pretrained_weights_dir = '/Users/ahmedtaha/Documents/Model/'

training_models_dir = './trained_models'
tensorbaord_dir = './tb/'

dataset_name = 'cars'
model = 'densenet161'


batch_size = 32
learning_rate = 0.1
end_learning_rate = 0
max_iter = 10000
logging_threshold = 100
caffe_iter_size = 12
suffix = '_train'

model_args = [dataset_name ,model ,'lr'+str(learning_rate),'B'+str(batch_size),suffix]
model_filename = '_'.join(model_args )
model_save_path = training_models_dir+model_filename
model_save_name = "patch_model.ckpt"
tensorbaord_file = os.path.basename(os.path.normpath(model_save_path)) #'20180304-180936'


if model == 'densenet161':
    network_name = 'nets.densenet161.DenseNet161'
    imagenet__weights_filepath = pretrained_weights_dir+'tf-densenet161/tf-densenet161.ckpt'
    preprocessing_module = 'data_sampling.augmentation.densenet_preprocessing'
    preprocess_func = 'densenet'


if dataset_name == 'cars':
    num_classes = 196
    db_path = dataset_dir+'stanford_cars'
    db_tuple_loader = 'data.cars_tuple_loader.CarsTupleLoader'
    train_csv_file = '/lists/train_all_sub_list.csv'
    val_csv_file = '/lists/val_all_sub_list.csv'
    test_csv_file = '/lists/test_all_sub_list.csv'

