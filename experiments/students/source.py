"""Source-side visual, motion and fusion initialization for the paper students."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from .cache import TensorCache, aligned_values, read_json
from .runtime import (CLASSES, build_student, load_checkpoint, seed_all, write,
                      move, model_forward, keep_cnn_eval)


class SourceView(Dataset):
    def __init__(self, base, stage, dataset):
        self.base, self.stage, self.dataset = base, stage, dataset
    def __len__(self):
        return len(self.base)
    def __getitem__(self, index):
        item = self.base[index]
        if self.stage == "motion" and self.dataset == "cuhkx":
            if item["teacher_features"].ndim == 3:
                item["teacher_features"] = item["teacher_features"].mean(1)
        if self.stage == "visual" and self.dataset == "mmact":
            item.update(time_position=item["global_time_position"],
                        teacher=F.softmax(item["teacher_logits"]/2, -1),
                        geometry=item["teacher_features"])
        return item


def local_mc3(visual, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    for name in ("stem", "layer1", "layer2", "layer3", "layer4"):
        prefix = name + "."
        values = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        getattr(visual, name).load_state_dict(values, strict=True)


def build_motion(dataset):
    from prism.student.native.mobind import NativeMoBindLite
    model = NativeMoBindLite(classes=CLASSES[dataset], width=96, alignment_width=64,
                            dropout=0.12, imu_instance_normalization=False,
                            imu_event_feature_width=0, skeleton_time_position=False)
    if dataset == "mmact":
        from prism.student.activity_imu import ActivityNativeIMUEncoder, NativeIMUSchema
        model.imu_encoder = ActivityNativeIMUEncoder(NativeIMUSchema.create(), width=96, dropout=0.12)
    elif dataset == "mmfit":
        from prism.student.workout_model import MMFitIMUEncoder
        model.imu_encoder = MMFitIMUEncoder()
    return model


def transfer_motion(unified, pretrained):
    for source, target in (
        (pretrained.skeleton_encoder, unified.motion_residual.encoder.skeleton_encoder),
        (pretrained.imu_encoder, unified.motion_residual.encoder.imu_encoder),
        (pretrained.skeleton_head, unified.motion_residual.semantic_head),
        (pretrained.skeleton_teacher_projection, unified.motion_residual.teacher_projection),
    ):
        target.load_state_dict(source.state_dict(), strict=True)


def input_dataset(args, *, validation=False):
    inputs = args.validation_inputs if validation else args.inputs
    label_path = args.validation_labels if validation else args.labels
    config = read_json(inputs)
    if config["dataset"] != args.dataset:
        raise ValueError("source manifest dataset mismatch")
    labels = aligned_values(label_path, [str(s) for s in config["record_id"]],
                            key="label", classes=CLASSES[args.dataset])
    base = TensorCache(inputs, cache_root=args.cache_root, source_labels=labels,
        source=True, raw=args.stage=="visual", motion_only=args.stage=="motion",
        augment=args.dataset=="cuhkx" and args.stage!="motion" and not validation,
        pixel_augment=args.stage=="visual", fold=args.fold)
    return SourceView(base, args.stage, args.dataset), labels


def save_model(path, model, **metadata):
    torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    **metadata), path)


def cuhkx_source(args, data, labels, device, out):
    config = read_json(Path(__file__).with_name(f"cuhkx_source_{args.stage}.json"))
    config.update(max_train_batches=0, max_eval_batches=0, live_visual_anchor=False)
    recipe = SimpleNamespace(**config)
    seed_all(20260811)
    if args.stage == "visual":
        from prism.student.native.visual import NativeMC3VisualStudent
        from .source_visual import train_epochs
        model = NativeMC3VisualStudent(frames=16, kinetics_pretrained=False, temporal_modeling=True)
        local_mc3(model, args.pretrained_weights)
        model.freeze_low_level("layer2")
        model.to(device)
        model, history, _ = train_epochs(model,
            DataLoader(data,batch_size=4,shuffle=True,num_workers=0,pin_memory=True),
            None, labels, recipe, device, epochs=16, early_stop=False)
    elif args.stage == "motion":
        from .source_motion import train
        model = build_motion("cuhkx").to(device)
        history = train(model, DataLoader(data,batch_size=64,shuffle=True,num_workers=0,
                                         pin_memory=True), labels, recipe, device)
    else:
        from .source_fusion import train_stage
        model = build_student("cuhkx")
        load_checkpoint(model.visual, args.visual_checkpoint)
        pretrained = build_motion("cuhkx")
        load_checkpoint(pretrained, args.motion_checkpoint)
        transfer_motion(model, pretrained)
        model.to(device)
        loader = DataLoader(data,batch_size=64,shuffle=True,num_workers=0,pin_memory=True)
        history = train_stage(model,loader,labels,recipe,device,"A",4)
        history += train_stage(model,loader,labels,recipe,device,"B",20)
    save_model(out/"final.pt",model,dataset=args.dataset,stage=args.stage)
    write(out/"history.json",history)


def class_weight(labels, classes, device):
    value = 1 / np.maximum(np.bincount(labels,minlength=classes),1)**0.35
    return torch.as_tensor(value/value.mean(),dtype=torch.float32,device=device)


def rgb_source(args, data, labels, validation, device, out):
    from .source_rgb_fusion import (set_stage, corrupt_visual_sequence, fusion_losses,
                                    visual_head_parameters, selective_anchor_loss)
    from .source_rgb_motion_losses import losses as motion_losses
    from .source_rgb_visual import (configure_visual_parameters, relation_gram_loss,
                                    _optimizer, _run_epoch, fixed_macro_f1)
    seed = 20260922 if args.dataset=="mmact" else 20260927
    seed_all(seed)
    weights = class_weight(labels,CLASSES[args.dataset],device)
    if args.dataset=="mmfit" and args.stage=="visual":
        workouts=data.base.workouts
        if "" in workouts:
            raise ValueError("MM-Fit visual source training requires workout metadata for stride-five sampling")
        chosen=[]
        for workout in dict.fromkeys(workouts):
            chosen.extend(np.flatnonzero(np.asarray(workouts)==workout)[::5])
        data=Subset(data,chosen)
    model = build_motion(args.dataset) if args.stage=="motion" else build_student(args.dataset)
    if args.stage=="visual":
        model = model.visual
        local_mc3(model,args.pretrained_weights)
    elif args.stage=="fusion":
        load_checkpoint(model.visual,args.visual_checkpoint)
        pretrained = build_motion(args.dataset)
        load_checkpoint(pretrained,args.motion_checkpoint)
        transfer_motion(model,pretrained)
    model.to(device)
    epochs = 16 if args.stage=="visual" and args.dataset=="mmact" else 2 if args.stage=="visual" else 24
    history=[]
    best=-1.0
    fixed_anchors = None
    if args.stage=="fusion" and args.dataset=="mmact":
        # The source visual anchor is frozen before either fusion stage.
        model.eval()
        fixed_anchors = torch.empty((len(data),CLASSES[args.dataset]))
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        with torch.no_grad():
            for batch in DataLoader(data,batch_size=64):
                rows=batch["_row_index"]
                batch=move(batch,device)
                fixed_anchors[rows]=model.visual.forward_from_backbone_sequence(
                    batch["backbone_sequence"],batch["view_valid"],batch["view_quality"],
                    batch["global_time_position"])["logits"].cpu()
        torch.set_rng_state(rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
    if args.stage=="visual" and args.dataset=="mmact":
        optimizer,scheduler=_optimizer(model)
    elif args.stage=="visual":
        backbone,head=configure_visual_parameters(model)
        optimizer=torch.optim.AdamW([dict(params=backbone,lr=1e-5),dict(params=head,lr=2e-4)],weight_decay=.05)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2,eta_min=1e-5)
    elif args.stage=="motion":
        optimizer=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=.05)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=24,
                              eta_min=2e-5 if args.dataset=="mmact" else 1e-5)
    def fusion_optimizer(stage):
        set_stage(model,stage)
        if args.dataset=="mmact":
            fresh={id(p) for p in model.motion_residual.encoder.fresh_parameters()}
            encoder=[p for p in model.motion_residual.encoder.parameters() if p.requires_grad and id(p) not in fresh]
            core={id(p) for p in encoder}
            fusion=[p for p in model.motion_residual.parameters() if p.requires_grad and id(p) not in core]
        else:
            encoder=list(model.motion_residual.encoder.parameters())
            fusion=[p for n,p in model.motion_residual.named_parameters() if not n.startswith("encoder.")]
        groups=[dict(params=encoder,lr=1e-4),dict(params=fusion,lr=4e-4)]
        if stage=="B" or args.dataset=="mmfit":
            groups.append(dict(params=visual_head_parameters(model.visual),lr=2e-5))
        opt=torch.optim.AdamW(groups,weight_decay=.05)
        return opt,torch.optim.lr_scheduler.CosineAnnealingLR(opt,
            T_max=(4 if stage=="A" else 20) if args.dataset=="mmact" else 24,eta_min=1e-5)
    if args.stage=="fusion":
        optimizer,scheduler=fusion_optimizer("A" if args.dataset=="mmact" else "B")
    scaler=torch.amp.GradScaler("cuda",enabled=args.stage=="visual" and args.dataset=="mmact" and device.type=="cuda")
    batch_size=4 if args.stage=="visual" and args.dataset=="mmact" else 2 if args.stage=="visual" else 64
    # MM-Fit keeps its own DataLoader generator across epochs.
    persistent=DataLoader(data,batch_size=batch_size,shuffle=args.stage!="visual",
                          generator=torch.Generator().manual_seed(seed),num_workers=0,pin_memory=True)
    for epoch in range(epochs):
        stage="A" if epoch<4 else "B"
        if args.stage=="fusion":
            if args.dataset=="mmact" and epoch==4:optimizer,scheduler=fusion_optimizer("B")
            else:set_stage(model,stage)
        loader=(DataLoader(data,batch_size=batch_size,shuffle=True,
                    generator=torch.Generator().manual_seed(seed+epoch),num_workers=0)
                if args.dataset=="mmact" else persistent)
        if args.stage=="visual" and args.dataset=="mmact":
            metrics=_run_epoch(model,loader,device,weights,optimizer,scaler,True,4,0)
        else:
            model.train()
            if args.stage=="fusion":
                keep_cnn_eval(model)
                if stage=="A":model.visual.eval()
            total=0.;seen=0;steps=0
            for batch in loader:
                batch=move(batch,device);optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,
                                    enabled=args.stage=="visual" and device.type=="cuda"):
                    if args.stage=="visual":
                        logits=model(batch["images"],batch["view_valid"],batch["view_quality"],batch["global_time_position"])["logits"]
                        loss=F.cross_entropy(logits,batch["label"],weight=weights,label_smoothing=.1)+F.kl_div(F.log_softmax(logits/2,-1),F.softmax(batch["teacher_logits"]/2,-1),reduction="batchmean")*4
                    elif args.stage=="motion":
                        values=motion_losses(model(batch,mask_ratio=.35),batch,class_weights=weights,imu_teacher_weight=1.)
                        loss=values["loss"]
                    else:
                        if fixed_anchors is not None:
                            anchor=fixed_anchors[batch["_row_index"].cpu()].to(device)
                        else:
                            with torch.no_grad():
                                anchor=model.visual.forward_from_backbone_sequence(batch["backbone_sequence"],batch["view_valid"],batch["view_quality"],batch["global_time_position"])["logits"]
                        sequence=corrupt_visual_sequence(batch["backbone_sequence"],batch["view_valid"])
                        output=model_forward(model,{**batch,"backbone_sequence":sequence})
                        loss=fusion_losses(output,{**batch,"anchor_logits":anchor},weights)["loss"]
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite source loss")
                loss.backward()
                active=[p for p in model.parameters() if p.requires_grad]
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in active):raise FloatingPointError("nonfinite source gradient")
                nn.utils.clip_grad_norm_(active,5. if args.dataset=="mmact" else 1.)
                optimizer.step();steps+=1;seen+=len(batch["label"]);total+=float(loss.detach())*len(batch["label"])
            metrics=dict(loss=total/seen,rows=seen,optimizer_updates=steps)
        scheduler.step()
        record=dict(epoch=epoch+1,train=metrics)
        if validation is not None:
            valdata,vallabels=validation
            if args.stage=="visual" and args.dataset=="mmact":
                result=_run_epoch(model,DataLoader(valdata,batch_size=4),device,weights,optimizer,scaler,False,4,0)
                score=result["macro_f1_fixed37"]
            else:
                model.eval();pred=[]
                with torch.no_grad():
                    for batch in DataLoader(valdata,batch_size=64):
                        batch=move(batch,device)
                        if args.stage=="motion":
                            value=model(batch,mask_ratio=0.)
                            pred.append(torch.stack((value["skeleton_logits"].argmax(1),value["imu_logits"].argmax(1)),1).cpu().numpy())
                        else:pred.append(model_forward(model,batch)["logits"].argmax(1).cpu().numpy())
                prediction=np.concatenate(pred)
                def f1(p):
                    return float(np.mean([2*np.sum((vallabels==c)&(p==c))/max(np.sum(vallabels==c)+np.sum(p==c),1) for c in range(CLASSES[args.dataset])]))
                score=.5*(f1(prediction[:,0])+f1(prediction[:,1])) if prediction.ndim==2 else f1(prediction)
                result=dict(macro_f1=score)
            record["validation"]=result
            if score>best:
                best=score;save_model(out/"best.pt",model,dataset=args.dataset,stage=args.stage,epoch=epoch+1)
        history.append(record)
        print(json.dumps(record),flush=True)
        save_model(out/"final.pt",model,dataset=args.dataset,stage=args.stage,epoch=epoch+1)
        write(out/"history.json",history)




def run_source(args):
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError("source output directory must be empty")
    args.output.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    data,labels=input_dataset(args)
    validation=input_dataset(args,validation=True) if args.validation_inputs else None
    if args.dataset=="cuhkx":cuhkx_source(args,data,labels,device,args.output)
    else:rgb_source(args,data,labels,validation,device,args.output)
