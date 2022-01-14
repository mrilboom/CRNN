import alphabets

proj_name = 'crnn_mobilenetV3'
raw_folder = ''
train_data = r'/root/Design/DATA'
test_data = r'/root/Design/DATA'
pretrain_model = r''
output_dir = './crnnout'
random_sample = True
random_seed = 3484
using_cuda = True
keep_ratio = False
gpu_id = '0'
data_worker = 0
batch_size = 32
img_height = 32
img_width = 160
alphabet = alphabets.alphabet
epoch = 200
display_interval = 10
save_interval = 100
test_interval = 100
test_disp = 160
test_batch_num = 16
lr = 0.0001
beta1 = 0.5
infer_img_w = 160
