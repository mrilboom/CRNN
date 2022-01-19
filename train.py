#coding=utf-8
import time
import shutil

import Config
import random
import os
import numpy as np
import torch
from torch.nn import CTCLoss
import torch.backends.cudnn as cudnn
import lib.dataset
import lib.convert
import lib.utility
from torch.autograd import Variable
import Net.net_new as Net
import torch.optim as optim

def val(net, da, criterion, writer, global_step, max_iter=30):
    print('Start val')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        da, shuffle=True, batch_size=Config.test_batch_num, num_workers=int(Config.data_worker))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = lib.utility.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        cpu_images, cpu_texts = data
        if i == max_iter - 1:
            show_image = cpu_images[-1].cpu().clone()
            show_image = show_image / 2 + 0.5  # unnormalize
            show_image = show_image.numpy()
            show_image = np.transpose(show_image, (1, 2, 0))
            import cv2
            show_image = cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB)
            # from matplotlib import pyplot as plt
            # plt.imshow(show_image)
            # plt.show()
        batch_size = cpu_images.size(0)
        lib.dataset.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        lib.dataset.loadData(text, t)
        lib.dataset.loadData(length, l)

        preds = net(image)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        # preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        list_1 = []
        for i in cpu_texts:
            list_1.append(i.decode('utf-8','strict'))
        #print(sim_preds)
        for pred, target in zip(sim_preds, list_1):
            if pred == target:
                n_correct += 1
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:Config.test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt.decode()))

    accuracy = n_correct / float(max_iter * Config.test_batch_num)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    writer.add_scalar('EVAL/acc', accuracy, global_step)
    writer.add_scalar('EVAL/loss', loss_avg.val(), global_step)
    if accuracy>0.5:
        writer.add_image(f"EVAL RESULTS/acc:{accuracy:.2g} - {sim_preds[-1]}", show_image, global_step,dataformats='HWC')

    return accuracy

def trainBatch(net, criterion, optimizer, train_iter):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    lib.dataset.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    lib.dataset.loadData(text, t)
    lib.dataset.loadData(length, l)

    preds = net(image)
    #print("preds.size=%s" % preds.size)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))  # preds.size(0)=w=22
    cost = criterion(preds, text, preds_size, length) / batch_size  # length= a list that contains the len of text label in a batch
    net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


if __name__ == '__main__':

    print("image scale: [%s,%s]\ngpu_id: %s\nbatch_size: %s" %
          (Config.img_height, Config.img_width, Config.gpu_id, Config.batch_size))

    random.seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    torch.manual_seed(Config.random_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = Config.gpu_id

    cudnn.benchmark = True
    if torch.cuda.is_available() and Config.using_cuda:
        cuda = True
        print('Using cuda')
    else:
        cuda = False
        print('Using cpu mode')

    train_dataset = lib.dataset.lmdbDataset(root=Config.train_data,is_training=True)
    test_dataset = lib.dataset.lmdbDataset(root=Config.test_data, transform=lib.dataset.resizeNormalize((Config.img_width, Config.img_height)))
    assert train_dataset
    print(train_dataset)
    # images will be resize to 32*100
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.batch_size,
        shuffle=True,
        num_workers=int(Config.data_worker),
        collate_fn=lib.dataset.alignCollate(imgH=Config.img_height, imgW=Config.img_width))

    n_class = len(Config.alphabet) + 1  # for python3
    #n_class = len(Config.alphabet.decode('utf-8')) + 1  # for python2
    print("alphabet class num is %s" % n_class)

    converter = lib.convert.strLabelConverter(Config.alphabet)
    #converter = lib.convert.StrConverter(Config.alphabet)
    # print(converter.dict)

    criterion = CTCLoss()

    net = Net.CRNN(n_class)
    
    # net.apply(lib.utility.weights_init)
    
    acc = 0
    acc_best = 0.5
    global_step = 0
    epoch = 0
    current_step=0
    if Config.pretrain_model:
        checkpoint = torch.load(Config.pretrain_model)
        net.load_state_dict(checkpoint['state_dict'])
        if not Config.finetune:
            global_step = checkpoint['global_step']
            epoch = checkpoint['epoch']
            acc_best = checkpoint['acc_best']
            current_step = checkpoint['current_step']

    image = torch.FloatTensor(Config.batch_size, 3, Config.img_height, Config.img_width)
    text = torch.IntTensor(Config.batch_size * 5)
    length = torch.IntTensor(Config.batch_size)

    if cuda:
        net.cuda()
        image = image.cuda()
        criterion = criterion.cuda()

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    loss_avg = lib.utility.averager()

    # optimizer = optim.RMSprop(net.parameters(), lr=Config.lr)
    #optimizer = optim.Adadelta(net.parameters(), lr=Config.lr)
    optimizer = optim.Adam(net.parameters(), lr=Config.lr,
                           betas=(Config.beta1, 0.999))
    #######

    # tensorboard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(Config.output_dir+"/"+Config.proj_name)
    try:
        # add graph
        in_channels = 1
        dummy_input = torch.zeros(1, in_channels, 32, 160, device='cuda')
        writer.add_graph(net, dummy_input)
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'{e}\n add graph to tensorboard failed')
    for epoch in range(epoch,Config.epoch):
        train_iter = iter(train_loader)
        i = current_step
        t0 = time.time()
        while i < len(train_loader):
            for p in net.parameters():
                p.requires_grad = True
            net.train()
            global_step+=1
            cost = trainBatch(net, criterion, optimizer, train_iter)
            loss_avg.add(cost)
            i += 1

            if i % Config.display_interval == 0:
                print(f'[{epoch}/{Config.epoch}][{i}/{len(train_loader)}] Loss: {loss_avg.val()}    '
                      f'next val eta--{int((Config.test_interval - i % Config.test_interval) / Config.display_interval * (time.time() - t0))}s    '
                      f'epoch eta --{int((len(train_loader) - i) / Config.display_interval * (time.time() - t0))}s   '
                      f'global_step:{global_step}')
                t0 = time.time()
              # tensorboard
                writer.add_scalar('TRAIN/LOSS', loss_avg.val(), global_step)
                # 学习率策略
                # writer.add_scalar('TRAIN/lr', lr, epoch)
                loss_avg.reset()

            if i % Config.test_interval == 0:
                acc = val(net, test_dataset, criterion,writer,global_step)
                t0 = time.time()
            # do checkpointing
            if i % Config.save_interval == 0:
                state = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'current_step': global_step,
                    'state_dict': net.state_dict(),
                    'acc_best': acc_best,
                }
                torch.save(state, f'{Config.output_dir}/{Config.proj_name}/model_current.pth')
                if acc>acc_best:
                    acc_best = acc
                    shutil.copyfile(f'{Config.output_dir}/{Config.proj_name}/model_current.pth', f'{Config.output_dir}/{Config.proj_name}/model_best.pth')
                

