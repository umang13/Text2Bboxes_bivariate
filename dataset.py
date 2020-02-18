import torch
import nltk
import pickle
from pycocotools.coco import COCO
import torch.utils.data as data
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
from sentence_transformers import SentenceTransformer

class COCODataset(data.Dataset) :

    def __init__(self,captionAnnFile, detectionAnnFile) :
        self.coco_captions = COCO(captionAnnFile)
        self.coco_detections = COCO(detectionAnnFile)
        self.ids = list(self.coco_captions.getAnnIds())
        self.embed_model = SentenceTransformer('bert-base-nli-mean-tokens')

    def __getitem__(self, index) :
        caption_id = self.ids[index]
        img_id = self.coco_captions.loadAnns(caption_id)[0]['image_id']
        det_ann_ids = self.coco_detections.getAnnIds(imgIds = img_id)
        caption_txt, caption_embedding = self.embed_caption(caption_id)
        bboxes, cats = self.retreive_bboxes_cats(det_ann_ids)
        return [caption_txt, caption_embedding], bboxes, cats
    
    def __len__(self) :
        return len(self.ids)
    
    def embed_caption(self, caption_id) :
        """Using pre trained vocabulary from BERT
           TODO : May want to shift to USE from Google"""
        caption = self.coco_captions.loadAnns(caption_id)
        caption_txt = caption[0]['caption']
        embeddings = self.embed_model.encode([caption[0]['caption']])
        return caption_txt, torch.Tensor(embeddings[0])
    
    def retreive_bboxes_cats(self, det_ann_ids) :
        det_anns = self.coco_detections.loadAnns(det_ann_ids)
        labels = self.get_category_labels()
        start_idx = labels['start']
        bboxes = []
        cats = [start_idx]

        for item in det_anns :
            if(not item['iscrowd']) :
                bboxes.append(item['bbox'])
                cats.append(item['category_id'])
        end_idx = labels['end']
        cats.append(end_idx)
        bboxes = torch.Tensor(bboxes)
        cats = torch.Tensor(cats)
        return bboxes, cats
    
    def get_category_labels(self) :
        categories = self.coco_detections.loadCats(self.coco_detections.getCatIds())
        labels = {}

        labels['pad'] = 0
        for category in categories :
            labels[category['name']] = category['id']
        max_idx = max(labels, key=labels.get)
        labels['start'] = labels[max_idx] + 1
        labels['end'] = labels[max_idx] + 2
        return labels

    def collate_fn(self, batch) :
        embeddings = list()
        captions = list()
        bboxes = list()
        cats = list()
        lengths = [len(b[2]) for b in batch]
        max_length = max(lengths)
        padded_box = torch.Tensor([0,0,0,0])
        for b in batch :
            captions.append(b[0][0])
            embeddings.append(b[0][1])
            padding = padded_box.repeat(max_length - b[1].shape[0], 1)
            bboxes.append(torch.cat((b[1], padding), dim=0))
            cats.append(b[2])
                
#         for bbox in bboxes :
            
       # bboxes_padded = torch.nn.utils.rnn.pad_sequence(bboxes)
        bboxes = torch.stack(bboxes, dim=0)
        cats_padded = torch.nn.utils.rnn.pad_sequence(cats, batch_first=True, padding_value=0)

        embeddings = torch.stack(embeddings, dim=0)
        return captions, embeddings, bboxes, cats_padded, lengths
        # collate_old = torch.utils.data.dataloader.default_collate
        # torch.utils.data.dataloader.default_collate = lambda batch: batch if all(map(torch.is_tensor, batch)) and any([tensor.size() != batch[0].size() for tensor in batch]) else collate_old(batch)
        # return torch.utils.data.dataloader.default_collate(batch)
    
        
def get_data_loader_and_cats(ann_caption, ann_detection, batch_size, shuffle) :
    coco = COCODataset(ann_caption ,ann_detection)
    data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=batch_size, shuffle=shuffle, collate_fn=coco.collate_fn, drop_last=True)

    labels = coco.get_category_labels()
    return data_loader, labels
    
        
# def get_data_loader_and_cats(ann_caption, ann_detection, batch_size, shuffle) :
#     coco = COCODataset(ann_caption ,ann_detection)
#     data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=batch_size, shuffle=shuffle)
#     labels = coco.get_category_labels()
#     return data_loader, labels
