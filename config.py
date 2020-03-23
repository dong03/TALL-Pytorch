import torch

batch_size = 56
DTYPE = torch.float32
debug = 0
if debug:
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
dropout_rate = 0.6
#train and test data path, replace it with your path
train_csv_path = "./exp_data/TACoS/train_clip-sentvec.pkl"
test_csv_path = "./TALL/exp_data/TACoS/test_clip-sentvec.pkl"
test_visual_feature_dir="/data/Dataset/Interval/Interval128_256_overlap0.8_c3d_fc6/"
train_visual_feature_dir = "/data/Dataset/Interval/Interval64_128_256_512_overlap0.8_c3d_fc6/"

lr = 0.001
EPOCH = 35000

tensoboard_logdir =' '
nname = ' '  #model name
model_pth = '/data/model/tall/%s.pth'%nname  #model path

#path to save some dataset output, can be loaded directly when used again
#can be any path
pth_clip_sentence_pairs_iou = ' '
