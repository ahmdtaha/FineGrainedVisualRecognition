
class BaseTupleLoader:

    def __init__(self):
        pass


    def imgs_and_lbls(self):
        images = []
        lbls = []
        imgs = self.data_df
        for img_idx in range(imgs.shape[0]):
            img_path = self.img_path + imgs.iloc[img_idx]['file_name']
            lbl = self.lbl2idx_dict[imgs.iloc[img_idx]['label']]
            images.append(img_path)
            lbls.append(lbl)

        return images, lbls
