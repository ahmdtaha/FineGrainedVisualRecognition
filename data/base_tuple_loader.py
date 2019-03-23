import numpy as np
import imageio
import cv2
import constants as const
import pandas as pd

class BaseTupleLoader:

    def __init__(self,args):
        csv_file = args['csv_file']
        db_path = args['db_path']
        self.data_df = pd.read_csv(db_path + csv_file)

        shuffle_data = args['shuffle'] if 'shuffle' in args else True
        if shuffle_data:
            self.data_permutation = np.random.permutation(self.data_df.shape[0])
        else:
            self.data_permutation = list(range(self.data_df.shape[0]))



    def imgs_and_lbls(self):

        """
        This functions returns a dataset, of images and labels, defined in the child sub-class inheriting this base-class .
        Args:
            No Args

        Returns:
            This function returns two lists
            The first and second list contains images and their corresponding labels respectively.

        Raises:
            No exceptions raised.
        """

        imgs = self.data_df
        images = imgs['file_name'].tolist()
        lbls = imgs['label'].tolist()
        for img_idx in range(imgs.shape[0]):
            images[img_idx] = self.img_path + images[img_idx]
            lbls[img_idx] = self.lbl2idx_dict[lbls[img_idx]]

        return images, lbls
