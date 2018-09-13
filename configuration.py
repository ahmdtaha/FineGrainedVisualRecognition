
dataset_name = 'cars'
model = 'densenet161'


batch_size = 32
learning_rate = 0.1
end_learning_rate = 0
max_iter = 40000
logging_threshold = 100
caffe_iter_size = 12



dataset_dir = '/Users/ahmedtaha/Documents/dataset/'
pretrained_weights_dir = '/Users/ahmedtaha/Documents/Model/'

if model == 'densenet161':
    network_name = 'nets.densenet161.DenseNet161'
    imagenet__weights_filepath = pretrained_weights_dir+'tf-densenet161/tf-densenet161.ckpt'
    preprocessing_module = 'data_sampling.augmentation.densenet_preprocessing'

if dataset_name == 'cars':
    num_classes = 196
    db_path = dataset_dir+'stanford_cars'
    db_tuple_loader = 'data_sampling.cars_tuple_loader.CarsTupleLoader'
    train_csv_file = '/lists/train_all_sub_list.csv'
    val_csv_file = '/lists/val_all_sub_list.csv'
    test_csv_file = '/lists/test_all_sub_list.csv'

