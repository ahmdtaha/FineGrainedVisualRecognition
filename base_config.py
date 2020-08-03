import argparse
import getpass
import os

# import constants as const


class BaseConfig:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--emb_dim', type=int, default=256,
                                 help='Embedding dimension')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='Batch Size')
        self.parser.add_argument('--gpu', type=str, default='0',
                                 help='which gpu')
        self.parser.add_argument('--checkpoint_dir', type=str, default=None,
                                 help='where to save experiment log and model')
        self.parser.add_argument('--db_name', type=str, default='flowers',
                                 help='Database name')
        self.parser.add_argument('--net', type=str, default='resnet50',
                                 help='Which networks? resnet50, inc4,densenet161')
        self.parser.add_argument('--tuple_loader_queue_size', type=int, default=10,
                                 help='Number of parallel threads')
        self.parser.add_argument('--triplet_loss_lambda', type=int, default=1,
                                 help='What is the balancing variable for triplet loss?')
        self.parser.add_argument('--caffe_iter_size', type=int, default=1,
                                 help='Aggregate gradient, like-caffe, across how many interations?')
        self.parser.add_argument('--logging_threshold', type=int, default=500,
                                 help='When to print the training loss log? (in iterations)')
        self.parser.add_argument('--test_interval', type=int, default=10,
                                 help='When to run eval on test split? after how many training iterations')
        self.parser.add_argument('--train_iters', type=int, default=40000,
                                 help='The number of training iterations')


        self.parser.add_argument('--Triplet_K', type=int, default=4,
                                 help='Number of samples per class in a mini-batch?')

        self.parser.add_argument('--checkpoint_filename', type=str, default='model.ckpt',
                                 help='what is the name of the checkpoint file?')
        self.parser.add_argument('--checkpoint_suffix', type=str, default='base_config',
                                 help='What is the suffix of the checkpoint file?')
        self.parser.add_argument('--learning_rate', type=float, default=0.01,
                                 help='What is the learning rate?')
        self.parser.add_argument('--end_learning_rate', type=float, default=0,
                                 help='What is the end learning rate? used to craft a polynomial decay scheduler')

        self.parser.add_argument('--log_filename', type=str, default='logger',
                                 help='what is the name of the training log file')



    def _load_user_setup(self):
        username = getpass.getuser()
        if username == 'ahmedtaha':  ## Local Machine
            logging_threshold = 1
            local_datasets_dir = '/Users/ahmedtaha/Documents/dataset/'
            # local_datasets_dir = '/Volumes/Backup/datasets/'
            pretrained_weights_dir = '/Users/ahmedtaha/Documents/Model/'
            training_models_dir = '/Users/ahmedtaha/Documents/dataset/dump/'
            dump_path = '/Users/ahmedtaha/Documents/dataset/dump/'
            batch_size = 32
            caffe_iter_size = 1
            debug_mode = True
        elif username == 'ataha':  # Machine
            logging_threshold = 500
            batch_size = 32
            local_datasets_dir = '/mnt/work/datasets/'
            pretrained_weights_dir = local_datasets_dir + 'Model/'
            training_models_dir = '/mnt/work/code/data_bal/model/'
            caffe_iter_size = 10
            debug_mode = False
            dump_path = ''
        else:  ## Assuming 32 GB GPU
            ##TODO : Make sure to fill these values
            local_datasets_dir = ''  # where is the dataset
            pretrained_weights_dir = local_datasets_dir + 'Model/'  # where the imagenet pre-trained weight
            training_models_dir = ''  # where to save the trained models
            dump_path = ''  ## I use this to dump files during debuging. I ought not use it here
            logging_threshold = 500
            batch_size = 32
            caffe_iter_size = 12
            debug_mode = False

        return local_datasets_dir,pretrained_weights_dir,training_models_dir,dump_path,logging_threshold,batch_size,caffe_iter_size,debug_mode

    def parse(self,args):
        cfg = self.parser.parse_args(args)

        local_datasets_dir, pretrained_weights_dir, training_models_dir, dump_path, logging_threshold, batch_size, caffe_iter_size, debug_mode = self._load_user_setup()
        cfg.num_classes, cfg.db_path, cfg.db_tuple_loader, cfg.train_csv_file, cfg.val_csv_file, cfg.test_csv_file    = self.db_configuration(cfg.db_name,local_datasets_dir)
        cfg.network_name, cfg.imagenet__weights_filepath, cfg.preprocess_func = self._load_net_configuration(cfg.net,pretrained_weights_dir)

        if cfg.checkpoint_dir is None:
            checkpoint_dir = [cfg.db_name, cfg.net, 'lr' + str(cfg.learning_rate), 'B' + str(cfg.batch_size),
                              'caf' + str(cfg.caffe_iter_size), 'iter' + str(cfg.train_iters // 1000) + 'K',
                              'lambda' + str(cfg.triplet_loss_lambda),
                              cfg.checkpoint_suffix]
            checkpoint_dir = '_'.join(checkpoint_dir)
            cfg.checkpoint_dir = os.path.join(training_models_dir, checkpoint_dir)
        else:
            cfg.checkpoint_dir = os.path.join(training_models_dir,cfg.checkpoint_dir)


        cfg.test_interval = cfg.test_interval * cfg.logging_threshold

        return cfg

    def _load_net_configuration(self,model,pretrained_weights_dir):
        if model == 'resnet50':
            network_name = 'nets.resnet_v2.ResNet50'
            imagenet__weights_filepath = pretrained_weights_dir + 'resnet_v2_50/resnet_v2_50.ckpt'
            preprocess_func = 'inception_v1'
        elif model == 'resnet50_v1':
            network_name = 'nets.resnet_v1.ResNet50'
            imagenet__weights_filepath = pretrained_weights_dir + 'resnet_v1_50/resnet_v1_50.ckpt'
            preprocess_func = 'vgg'

        elif model == 'densenet161':
            network_name = 'nets.densenet161.DenseNet161'
            imagenet__weights_filepath = pretrained_weights_dir + 'tf-densenet161/tf-densenet161.ckpt'
            preprocess_func = 'densenet'

        elif model == 'inc4':
            network_name = 'nets.inception_v4.InceptionV4'
            imagenet__weights_filepath = pretrained_weights_dir + 'inception_v4/inception_v4.ckpt'
            preprocess_func = 'inception_v1'

        elif model == 'inc3':
            network_name = 'nets.inception_v3.InceptionV3'
            imagenet__weights_filepath = pretrained_weights_dir + 'inception_v3.ckpt'
            preprocess_func = 'inception_v1'

        elif model == 'mobile':
            network_name = 'nets.mobilenet_v1.MobileV1'
            imagenet__weights_filepath = pretrained_weights_dir + 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt'
            preprocess_func = 'inception_v1'

        else:
            raise NotImplementedError('network name not found')

        return network_name,imagenet__weights_filepath,preprocess_func

    def db_configuration(self, dataset_name, datasets_dir):

        if dataset_name == 'flowers':
            num_classes = 102
            db_path = datasets_dir + 'flower102'
            db_tuple_loader = 'data.flower_tuple_loader.FLower102TupleLower'
            train_csv_file = '/lists/train_all_sub_list.csv'
            val_csv_file = '/lists/val_all_sub_list.csv'
            test_csv_file = '/lists/test_all_sub_list.csv'
        elif dataset_name == 'cars':
            num_classes = 196
            db_path = datasets_dir + 'stanford_cars'
            db_tuple_loader = 'data.cars_tuple_loader.CarsTupleLoader'
            train_csv_file = '/lists/train_all_sub_list.csv'
            val_csv_file = '/lists/val_all_sub_list.csv'
            test_csv_file = '/lists/test_all_sub_list.csv'
        elif dataset_name == 'aircrafts':
            num_classes = 100
            db_path = datasets_dir + 'aircrafts'
            db_tuple_loader = 'data.aircrafts_tuple_loader.AircraftsTupleLoader'
            train_csv_file = '/lists/train_all_sub_list.csv'
            val_csv_file = '/lists/val_all_sub_list.csv'
            test_csv_file = '/lists/test_all_sub_list.csv'
        elif dataset_name == 'cub':
            num_classes = 200
            db_path = datasets_dir + 'CUB_200_2011'
            db_tuple_loader = 'data.CUB_tuple_loader.CUBTupleLoader'
            train_csv_file = '/lists/train_all_sub_list.csv'
            val_csv_file = '/lists/val_sub_list.csv'
            test_csv_file = '/lists/test_all_sub_list.csv'
        elif dataset_name == 'dogs':
            num_classes = 120
            db_path = datasets_dir + 'Stanford_dogs'
            db_tuple_loader = 'data.dogs_tuple_loader.DogsTupleLoader'
            train_csv_file = '/lists/train_all_sub_list.csv'
            val_csv_file = '/lists/val_sub_list.csv'
            test_csv_file = '/lists/test_all_sub_list.csv'
        elif dataset_name == 'birds':
            num_classes = 555
            db_path = datasets_dir + 'nabirds'
            db_tuple_loader = 'data.birds_tuple_loader.BirdsTupleLoader'
            train_csv_file = '/lists/train_all_sub_list.csv'
            val_csv_file = '/lists/val_sub_list.csv'
            test_csv_file = '/lists/test_all_sub_list.csv'
        else:
            raise NotImplementedError('dataset_name not found')

        return num_classes,db_path,db_tuple_loader,train_csv_file,val_csv_file,test_csv_file

