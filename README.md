#CDSE-UNet
The earlier version of this paper has been presented as arXiv in Cornell University according to the following link: 
       https://arxiv.org/abs/2403.01513.

#### soft enviroment
   torch1.8.1+cuda111

#### Instructions 

1. Dataset original image storage address: E:\pw_2024\PycharmProjects\data\coronacases\traindata   
   Label data storage address:E:\pw_2024\PycharmProjects\data\coronacases\valdata
   Note: Canny edge images are also stored in the label data

2.  mynet3_threshold.py: model
    data5.py:   make data
3. Run train.py directly, where the train_image folder stores the effect images during the training process
4. Save weights in the params folder






