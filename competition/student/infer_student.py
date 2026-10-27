"""Pure challenge Student forward pass over freshly-built caches."""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent; LEG=HERE/"modules"
sys.path.insert(0,str(LEG))
from deploy_model import build_deploy_model
from test_data import TestPixelDataset, TestCachedSequenceMotionDataset, collate_test
from cached_motion_data import collate_cached_motion
from train_mobind_fusion_proxy import model_forward, to_device

def main():
 started=time.perf_counter()
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--sequence-cache',type=Path,required=True); p.add_argument('--motion-cache',type=Path,required=True); p.add_argument('--pixel-cache',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--device',default='auto'); a=p.parse_args()
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 model=build_deploy_model(ck['deployment_model_config']); model.load_state_dict(ck['model_state'],strict=True)
 device=torch.device('cuda' if a.device=='cuda' or (a.device=='auto' and torch.cuda.is_available()) else 'cpu'); model.to(device).eval()
 pix=TestPixelDataset(a.pixel_cache); seq=a.sequence_cache.resolve(); seq.mkdir(parents=True,exist_ok=True)
 n=len(pix); frames=int(ck['deployment_model_config']['visual']['frames']); path=seq/'backbone_sequence_fp16.npy'; done=seq/'completed.npy'
 arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(n,2,3,frames,512)); fin=np.lib.format.open_memmap(done,mode='w+',dtype=np.uint8,shape=(n,)); arr[:]=0; fin[:]=0
 loader=DataLoader(pix,batch_size=a.batch_size,shuffle=False,num_workers=0,collate_fn=collate_test)
 with torch.inference_mode():
  for batch in loader:
   b=to_device(batch,device)
   with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'): z=model.visual.encode_backbone_sequence(b['images'])
   idx=np.asarray(batch['cache_index'],dtype=np.int64); arr[idx]=z.float().cpu().numpy().astype(np.float16); fin[idx]=1
 arr.flush(); fin.flush()
 rows=[{'sample_id':x['sample_id'],'source_id':x['source_id'],'user_id':'anonymous','class_id':'-1'} for x in pix.rows]
 with (seq/'rows.csv').open('w',encoding='utf-8-sig',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 ds=TestCachedSequenceMotionDataset(seq,a.motion_cache,a.pixel_cache); loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=0,collate_fn=collate_cached_motion)
 out=[]
 with torch.inference_mode():
  for batch in loader:
   b=to_device(batch,device)
   with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
    logits=model_forward(model,b)['logits']
   logits=logits.float().cpu()
   if not torch.isfinite(logits).all(): raise RuntimeError('non-finite Student logits')
   pred=logits.argmax(1).tolist(); out.extend(zip(batch['sample_id'],pred))
 with a.output.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f); w.writerow(('path','prediction'))
  manifest_rows=list(csv.DictReader(a.manifest.open(encoding='utf-8-sig')))
  predicted=dict(out)
  if len(out) != len(manifest_rows) or set(predicted) != {r['official_sample_id'] for r in manifest_rows}:
   raise RuntimeError('prediction row count or IDs differ from manifest')
  for row in manifest_rows: w.writerow((row['official_path'],predicted[row['official_sample_id']]))
 metrics={'device':str(device),'rows':n,'student_seconds_including_load':time.perf_counter()-started,
          'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
 (a.output.parent/'inference_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': main()
