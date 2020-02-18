import torch
from pycocotools.coco import COCO
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys
from model_new import Text2BBoxesModel
from dataset_bucketized import get_data_loader_and_cats
from torch import autograd
# CUDA_LAUNCH_BLOCKING=1
dataType='train2017'
dataDir = './Datasets/coco/images/{}/'.format(dataType)
annFile_Detection ='./Datasets/coco/annotations/instances_{}.json'.format(dataType)
annFile_Caption ='./Datasets/coco/annotations/captions_{}.json'.format(dataType)
batch_size = 64
coco = COCO(annFile_Detection)
coco_captions = COCO(annFile_Caption)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_size = 768
clip_value = 10
epochs = 20
learning_rate = 1.2e-7
beta1 = 0.9
beta2 = 0.999
max_bucket_size = 5
lambda_where = 1
lambda_what = 4
gamma = 0.5
mu_xy = torch.Tensor([82.22,72.81])
mu_wh = torch.Tensor([83.97,81.61])
sigma_xy = torch.Tensor([137.50, 116.07])
sigma_wh = torch.Tensor([143.86, 134.05])
ckpt_file_path = "./outputs/checkpoint_w_bboxes_bucketized_fix10.ckpt"
ckpt_file_path_save = "./outputs/checkpoint_w_bboxes_bucketized_fix10_cont.ckpt"
print(device)


dataloader, total_categories = get_data_loader_and_cats(annFile_Caption, annFile_Detection, batch_size, True, max_bucket_size)

num_classes = total_categories[max(total_categories, key=total_categories.get)] + 1
model = Text2BBoxesModel(embed_size, total_categories, batch_size, device)
model = model.to(device)
mu_xy = mu_xy.to(device)
mu_wh = mu_wh.to(device)
sigma_xy = sigma_xy.to(device)
sigma_wh = sigma_wh.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas=(beta1, beta2))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
if(os.path.exists(ckpt_file_path)) :
    checkpoint = torch.load(ckpt_file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    count = checkpoint['count']
    print("Model loaded with training up to {} epochs and {} minibatches".format(epoch, count))
else :
    print("Training model afresh!")
# with autograd.detect_anomaly():
for epoch in range(epochs) : 
    count = 0
    running_loss = 0
    running_loss_what = 0
    running_loss_where = 0
    model.train()
    for _, captions_mini_batch, bboxes, categories, lengths in dataloader :
        captions_mini_batch = captions_mini_batch.to(device)
        bboxes = bboxes.to(device)
        categories = categories.to(device)
        loss_final = 0
        loss_what = 0
        loss_where = 0
        loss_where_coords = 0
        loss_where_size = 0
        optimizer.zero_grad()
        max_length = max(lengths)
        lengths = torch.Tensor(lengths)
        outputs = model(captions_mini_batch, max_length)

        categories = categories.permute(1, 0)
        seqlen = len(outputs)
        for i in range(0,seqlen) :
            pred_labels = outputs[i][0]
            theta_xy = outputs[i][1]
            theta_wh = outputs[i][2]

            loss_what += F.cross_entropy(pred_labels, categories[i, :].long())
            loss_where_coords += model.mdn_loss(theta_xy[0], theta_xy[1], theta_xy[2], (bboxes[:,i,:2] - mu_xy)/sigma_xy) #+ model.mdn_loss(theta_wh[0], theta_wh[1], theta_wh[2], bboxes[:,i,2:])
            loss_where_size += model.mdn_loss(theta_wh[0], theta_wh[1], theta_wh[2], (bboxes[:,i,2:] - mu_wh)/sigma_wh)
        loss_where = loss_where_size + loss_where_coords
        loss_final = lambda_where * loss_where + lambda_what * loss_what
        loss_final /= seqlen
        loss_final.backward()
        # print(loss_final)
        # for name, param in model.named_parameters():
        #     if not param.requires_grad:
        #         print(name, param.data)
        # print("======> " , loss_final)
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
        optimizer.step()
        count += 1  
        running_loss_what += loss_what/seqlen
        running_loss_where += loss_where/seqlen  
        if(count % 100 == 0) :
            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict' : optimizer.state_dict(),
                          'scheduler_state_dict' : scheduler.state_dict(),
                          'epoch'  : epoch,
                          'count'  : count,
                          'loss_final' : loss_final}
            torch.save(checkpoint, ckpt_file_path_save)
            print("After processing {} minibatches in epoch {} , loss is {}".format(count, epoch, loss_final), flush=True)
            print(loss_what/seqlen, loss_where_coords/seqlen, loss_where_size/seqlen, flush=True)
    running_loss = lambda_what * running_loss_what + lambda_where * running_loss_where
    print("After epoch {} :\n  loss on labels {} , loss on bboxes {} , total loss {}".format(epoch, running_loss_what/count, running_loss_where/count, running_loss/count), flush=True)
    scheduler.step()


    print("Predictions on validation sets")
    val_mu_xy = torch.Tensor([82.22,72.81])
    val_mu_wh = torch.Tensor([83.97,81.61])
    val_sigma_xy = torch.Tensor([137.50, 116.07])
    val_sigma_wh = torch.Tensor([143.86, 134.05])
    with torch.no_grad() :
        model.eval()
        dataType = 'val2017'
        dataDir = './Datasets/coco/images/{}/'.format(dataType)
        annFile_Detection ='./Datasets/coco/annotations/instances_{}.json'.format(dataType)
        annFile_Caption ='./Datasets/coco/annotations/captions_{}.json'.format(dataType)
        val_dataloader, total_categories = get_data_loader_and_cats(annFile_Caption, annFile_Detection, batch_size, True, max_bucket_size)
        for caption_txt, caption, bboxes, categories, lengths in val_dataloader :
            val_loss = 0
            val_loss_what = 0
            val_loss_where = 0
            caption = caption.to(device)
            bboxes = bboxes.to(device)
            categories = categories.to(device)
            max_length = max(lengths)
            val_outs = model(caption, max_length)
            categories = categories.permute(1, 0)
            seqlen = len(val_outs) 
            pred_seq = []
            pred_bbox_coords = []
            pred_bbox_size = []
            for i in range(0,seqlen) :
                pred_labels = val_outs[i][0]
                theta_xy = val_outs[i][1]
                theta_wh = val_outs[i][2]
                pred_seq.append(torch.max(pred_labels, 1)[1])
                pred_bbox_coords.append(model.sample_from_gaussian(theta_xy))
                pred_bbox_size.append(model.sample_from_gaussian(theta_wh))
                val_loss_what += F.cross_entropy(pred_labels, categories[i, :].long())
                val_loss_where += model.mdn_loss(theta_xy[0], theta_xy[1], theta_xy[2], (bboxes[:,i,:2] - mu_xy)/sigma_xy) + model.mdn_loss(theta_wh[0], theta_wh[1], theta_wh[2], (bboxes[:,i,2:] - mu_wh)/sigma_wh)
            val_loss = lambda_what * val_loss_what + lambda_where * val_loss_where
            val_loss /= seqlen

        print("Loss on validaton set : {}".format(val_loss))
        key_list = list(total_categories.keys()) 
        val_list = list(total_categories.values()) 

        # Print some validation set inputs and outputs 
        num_samples = 10
        for i in range(num_samples) :
            print("Text input : {}".format(caption_txt[i]))
            print("Ground truth labels: {} , ground truth bbox coords : {} , ground truth bbox sizes :{}".format([key_list[val_list.index(j)] for j in categories[:,i].tolist()], bboxes[i,:,:2], bboxes[i,:,2:]))
            label = []
            bbox_coor = []
            bbox_sizes = []
            for j in range(len(pred_seq)) :
                label.append(pred_seq[j][i])
                bbox_coor.append(pred_bbox_coords[j][i] * val_sigma_xy + val_mu_xy)
                bbox_sizes.append(pred_bbox_size[j][i] * val_sigma_wh + val_mu_wh)
            label = torch.stack(label, dim = 0)
            bbox_coor = torch.stack(bbox_coor, dim=0)
            bbox_sizes = torch.stack(bbox_sizes, dim=0)
            print("Predicted labels : {} , predicted bbox x,y coords : {} , predicted bbox w,h sizes : {}".format([key_list[val_list.index(j)] for j in label], bbox_coor, bbox_sizes))

