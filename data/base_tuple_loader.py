import configuration as config
import numpy as np
import imageio
import cv2
import constants as const
import pandas as pd

class BaseTupleLoader:

    def __init__(self,args):
        csv_file = args['csv_file']
        db_path = config.db_path
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

        images = []
        lbls = []
        imgs = self.data_df
        for img_idx in range(imgs.shape[0]):
            img_path = self.img_path + imgs.iloc[img_idx]['file_name']
            lbl = self.lbl2idx_dict[imgs.iloc[img_idx]['label']]
            images.append(img_path)
            lbls.append(lbl)

        return images, lbls
