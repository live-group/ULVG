import clip

import torch
import torch.nn as nn
import time
import numpy as np
import copy
import os

from detectron2.data import MetadataCatalog

class localClip(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        device=cfg.MODEL.DEVICE
        self.model, self.preprocess = clip.load('RN101', device)
        self.model.float()
        # freeze everything
        for name, val in self.model.named_parameters():
            val.requires_grad = False
        self.enc = self.model.visual

        domain_prompt_wo = []
        domain_prompt_w = []
        with open(cfg.DOMAIN_AUG_FILE,'r') as f:
            for ind,l in enumerate(f):
                domain_prompt_wo.append(l.strip().split('--')[0])
                domain_prompt_w.append(l.strip().split('--')[1])
        with torch.no_grad():
            text_inputs_wo = torch.cat([clip.tokenize(prompt) for prompt in domain_prompt_wo]).to(device)
            text_features_wo = self.model.encode_text(text_inputs_wo).float()
            text_inputs_w = torch.cat([clip.tokenize(prompt) for prompt in domain_prompt_w]).to(device)
            text_features_w = self.model.encode_text(text_inputs_w).float()
        self.text_diff = text_features_w-text_features_wo
        self.text_diff /= self.text_diff.norm(dim=-1, keepdim=True)


        prompt_class_causal = []
        prompt_class_noncausal = []
        with open('class_description_discriminative_class.txt', 'r') as f:
            for ind, l in enumerate(f):
                prompt_class_causal.append(l.strip().split(';'))
        with open('class_description_spurious_class.txt', 'r') as f:
            for ind, l in enumerate(f):
                prompt_class_noncausal.append(l.strip().split(';'))
        with torch.no_grad():
            class_causal_text_inputs = torch.cat([clip.tokenize(prompt) for prompt in prompt_class_causal]).to(device)
            class_causal_text_features = self.model.encode_text(class_causal_text_inputs).float()
            class_noncausal_text_inputs = torch.cat([clip.tokenize(prompt) for prompt in prompt_class_noncausal]).to(device)
            class_noncausal_text_features = self.model.encode_text(class_noncausal_text_inputs).float()
        class_causal_text_features /= class_causal_text_features.norm(dim=-1, keepdim=True)
        class_noncausal_text_features /= class_noncausal_text_features.norm(dim=-1, keepdim=True)
        class_causal_text_vectors=class_causal_text_features
        Q, R = torch.linalg.qr(class_causal_text_vectors.T)
        self.class_causal_text_basisvectors = Q.T
        class_noncausal_text_vectors=class_noncausal_text_features
        Q, R = torch.linalg.qr(class_noncausal_text_vectors.T)
        self.class_noncausal_text_basisvectors = Q.T


    def forward(self, image):
        x = image
        x = self.enc.relu1(self.enc.bn1(self.enc.conv1(x)))
        x = self.enc.relu2(self.enc.bn2(self.enc.conv2(x)))
        x = self.enc.relu3(self.enc.bn3(self.enc.conv3(x)))
        x = self.enc.avgpool(x)

        x = self.enc.layer1(x)
        x = self.enc.layer2(x)
        x = self.enc.layer3(x)
        return {"res4": x}

                                            
    
    


    


   

    
