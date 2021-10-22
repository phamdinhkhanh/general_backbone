from general_backbone.data import AugmentationDataset

if __name__ == '__main__':
    augdataset = AugmentationDataset(data_dir='toydata/image_classification',
                            name_split='train',
                            config_file = 'general_backbone/configs/image_clf_config.py', 
                            dict_transform=None, 
                            input_size=(256, 256), 
                            debug=True, 
                            dir_debug = 'tmp/alb_img_debug', 
                            class_2_idx=None)

    for i in range(50):
        img, label = augdataset.__getitem__(i)
        print(img.size)


