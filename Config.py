import alphabets

raw_folder = ''
train_data = r'F:\Design\train'
test_data = r'F:\Design\test'
pretrain_model = r'C:\Users\mrilboom\PycharmProjects\Design\CRNN-master\crnnout\model_current.pth'
output_dir = './crnnout'
random_sample = True
random_seed = 3484
using_cuda = True
keep_ratio = False
gpu_id = '0'
model_dir = './models'
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
