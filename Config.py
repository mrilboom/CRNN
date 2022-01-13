import alphabets

raw_folder = ''
train_data = '/root/Design/Data'
test_data = '/root/Design/Data'
pretrain_model = ''
output_dir = ''
random_sample = True
random_seed = 3484
using_cuda = False
keep_ratio = False
gpu_id = '5'
model_dir = './w160_bs64_model'
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
