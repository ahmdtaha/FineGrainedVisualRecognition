from data.base_tuple_loader import BaseTupleLoader
import numpy as np
import pandas as pd
import configuration as config

class CarsTupleLoader(BaseTupleLoader):

    def __init__(self,args):
        BaseTupleLoader.__init__(self)
        csv_file = args['csv_file']
        self.data_df = pd.read_csv(config.db_path + csv_file)
        self.img_path = config.db_path + '/'

        lbls = self.data_df['label']
        lbl2idx = np.sort(np.unique(lbls))

        self.lbl2idx_dict = {k: v for v, k in enumerate(lbl2idx)}
        self.final_lbls = [self.lbl2idx_dict[x] for x in list(lbls.values)]

        self.num_classes = len(self.lbl2idx_dict.keys())
        self.data_permutation = np.random.permutation(self.data_df.shape[0])
        self.data_idx = 0


        print('Data size ', self.data_df.shape[0], 'Num lbls', len(self.lbl2idx_dict.keys()))
