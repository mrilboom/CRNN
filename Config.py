import alphabets

proj_name = 'crnn_mobilenetV3_h256'
raw_folder = ''
train_data = '/data/zjj/dataset/train'
test_data = '/data/zjj/dataset/test'
pretrain_model = r''
finetune = False
output_dir = './crnnout'
random_sample = True
random_seed = 1111
using_cuda = True
keep_ratio = False
gpu_id = '3'
data_worker = 4
batch_size = 256
img_height = 32
img_width = 160
alphabet = alphabets.alphabet
epoch = 20
display_interval = 10
save_interval = 100
test_interval = 100
test_disp = 10
test_batch_num = 32
lr = 0.0008
beta1 = 0.5
infer_img_w = 160