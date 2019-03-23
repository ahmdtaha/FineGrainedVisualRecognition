from data.base_tuple_loader import BaseTupleLoader
import numpy as np

class DogsTupleLoader(BaseTupleLoader):

    def __init__(self,args):
        BaseTupleLoader.__init__(self,args)

        db_path = args['db_path']
        self.img_path = db_path + '/Images/'
        self.lbls = self.data_df['label']
        lbl2idx = np.sort(np.unique(self.lbls))

        self.lbl2idx_dict = {k: v for v, k in enumerate(lbl2idx)}
        self.final_lbls = [self.lbl2idx_dict[x] for x in list(self.lbls.values)]
        self.num_classes = len(self.lbl2idx_dict.keys())
        print(self.__class__.__name__,' Data size ', self.data_df.shape[0], 'Num lbls', len(self.lbl2idx_dict.keys()))