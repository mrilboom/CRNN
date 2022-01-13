import alphabets

raw_folder = ''
train_data = '/root/Design/DATA'
test_data = '/root/Design/DATA'
pretrain_model = ''
output_dir = 'models/'
random_sample = True
random_seed = 3484
using_cuda = True
keep_ratio = False
gpu_id = '5'
model_dir = './models'
data_worker = 1
batch_size = 16
img_height = 32
img_width = 160
alphabet = alphabets.alphabet
epoch = 200
display_interval = 16
save_interval = 220
test_interval = 220
test_disp = 160
test_batch_num = 16
lr = 0.0001
beta1 = 0.5
infer_img_w = 160
