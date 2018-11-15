from data.base_tuple_loader import BaseTupleLoader
import numpy as np
import pandas as pd
import configuration as config

class FLower102TupleLower(BaseTupleLoader):

    def __init__(self,args=None):
        BaseTupleLoader.__init__(self,args)

        self.img_path = config.db_path + '/jpg/'

        lbls = self.data_df['label']
        lbl2idx = np.sort(np.unique(lbls))

        self.lbl2idx_dict = {k: v for v, k in enumerate(lbl2idx)}
        self.final_lbls = [self.lbl2idx_dict[x] for x in list(lbls.values)]

        self.num_classes = len(self.lbl2idx_dict.keys())
        self.data_idx = 0

        print('Data size ', self.data_df.shape[0], 'Num lbls', len(self.lbl2idx_dict.keys()))