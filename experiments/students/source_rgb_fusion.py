"""Original MMAct/MM-Fit source fusion objectives."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .source_fusion import visual_head_parameters
def set_stage(model,stage):
 for p in model.parameters():p.requires_grad_(False)
 for p in model.motion_residual.parameters():p.requires_grad_(True)
 if stage=='B':
  for p in visual_head_parameters(model.visual):p.requires_grad_(True)
 elif stage!='A':raise ValueError(stage)
 for module in (model.visual.stem,model.visual.layer1,model.visual.layer2,model.visual.layer3,model.visual.layer4):
  for p in module.parameters():p.requires_grad_(False)
  module.eval()

def keep_cnn_eval(model):
 for module in (model.visual.stem,model.visual.layer1,model.visual.layer2,model.visual.layer3,model.visual.layer4):module.eval()

def corrupt_visual_sequence(sequence,valid,*,probability=.75,feature_dropout=.3,view_dropout=.4):
 selected=torch.rand(len(sequence),1,1,1,1,device=sequence.device)<probability;features=torch.rand(len(sequence),1,1,1,sequence.shape[-1],device=sequence.device)>=feature_dropout;views=torch.rand(len(sequence),2,3,1,1,device=sequence.device)>=view_dropout;empty=~views.any(2,keepdim=True);fallback=torch.zeros_like(views);fallback[:,:,0]=True;views=torch.where(empty,fallback,views);corrupt=sequence*features*views;return torch.where(selected,corrupt,sequence)

def selective_anchor_loss(logits,anchor,labels,t=2.):
 valid=anchor.argmax(1)==labels
 if not valid.any():return logits.sum()*0
 per=torch.nn.functional.kl_div(torch.nn.functional.log_softmax(logits/t,-1),torch.nn.functional.softmax(anchor/t,-1),reduction='none').sum(1)*t*t;return per[valid].mean()

def relation_gram(embedding,teacher):
 a=nn.functional.normalize(embedding.float(),dim=1);b=nn.functional.normalize(teacher.float(),dim=1);return nn.functional.mse_loss(a@a.T,b@b.T)

def fusion_losses(out,batch,weights):
 ce=nn.functional.cross_entropy(out['logits'],batch['label'],weight=weights,label_smoothing=.08);kd=nn.functional.kl_div(nn.functional.log_softmax(out['logits']/2,-1),nn.functional.softmax(batch['teacher_logits']/2,-1),reduction='batchmean')*4;rel=relation_gram(out['visual_embedding'],batch['teacher_features']);motion=nn.functional.cross_entropy(out['motion_logits'],batch['label'],weight=weights,label_smoothing=.08);anchor=selective_anchor_loss(out['logits'],batch['anchor_logits'],batch['label']);total=ce+kd+.1*rel+.35*motion+.3*anchor;return {'loss':total,'ce':ce,'kd':kd,'relation':rel,'motion_aux':motion,'anchor':anchor}
