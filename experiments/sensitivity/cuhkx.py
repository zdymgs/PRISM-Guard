"""CUHK-X S1; construction copied from e1r_cuhkx_construct.py."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
sys.dont_write_bytecode = True
PRISM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PRISM))
from prism.core.sequence import RecordingMetadata, DecoderConfig, build_sessions, fit_transition_model, decode_sessions
from prism.eval.metrics import metrics
RUNS = ['p427_foundation_rebuild_v1','p428_skeleton_rebuild_v1','p429_thermal_rebuild_full_v1','p430_token_rebuild_full_v1','p431_irthermal_full_v2','p432_lavila_full_v1','p433_physical_full_v1']
PATH_NAMES = {"direct":"D", "group":"D→G", "sequence":"D→G→S", "direct_sequence":"D→S", "direct_sequence_nogate":"D→S (no gate)", "online_forward":"online_forward"}
FINAL = {"transition_lambda":0.45,"trigram_backoff":1.0,"session_gap_seconds":30.0}
GRID = {"transition_lambda":[0.25,0.35,0.45,0.55,0.65],"trigram_backoff":[0.5,1.0,2.0,5.0],"session_gap_seconds":[15.0,20.0,30.0,45.0,60.0]}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def align_metadata(path, sample_ids):
    table = pd.read_csv(path, usecols=['sample_id','user_id','recording_date','start_seconds'], dtype={'sample_id':str,'user_id':str,'recording_date':str}).fillna({'user_id':'','recording_date':''}).set_index('sample_id').loc[sample_ids]
    return RecordingMetadata(np.asarray(sample_ids).astype(str), table['user_id'].to_numpy(str), table['recording_date'].to_numpy(str), table['start_seconds'].to_numpy(float))

def stack_fit(x,y,weighted=False):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.03,solver='lbfgs',max_iter=1000,random_state=20260912))
    model.fit(x,y,logisticregression__sample_weight=np.where(x[:,-1]>0,2.,1.) if weighted else np.ones(len(x)))
    return model

def prob(model,x):
    p=np.zeros((len(x),40));p[:,model[-1].classes_.astype(int)]=model.predict_proba(x);return p

def crossfit(x,y,cv,weighted=False):
    p=np.zeros((len(x),40))
    for fit,held in cv:p[held]=prob(stack_fit(x[fit],y[fit],weighted),x[held])
    return p

def graph(ids,base,bank,metadata_path,session_gap_seconds,mode='aligned',seed=None):
    from prism._chains.repeats import CONFIG, expert_evidence as evidence
    from prism._chains.repeats import date_session_lists,cluster_sessions
    from prism.core.alignment import align_probabilities
    meta=align_metadata(metadata_path,ids);lookup={str(i):p for i,p in zip(ids,bank)};vote,_=evidence(ids,base,lookup,'vote');peers=[[] for _ in ids];rng=np.random.default_rng(seed);replacement_count=0
    for sessions in date_session_lists(np.arange(len(ids)),meta,session_gap_seconds):
        for group in cluster_sessions(sessions,vote,base,meta,CONFIG):
            ref=max(group,key=len)
            for other in group:
                if other is ref:continue
                pairs=list(zip(range(min(len(ref),len(other))),range(min(len(ref),len(other))))) if mode=='position' else align_probabilities(vote[ref],vote[other],CONFIG.alignment_gap_penalty)[0]
                for i,j in pairs:
                    left,right=int(ref[i]),int(other[j]);peers[left].append(right);peers[right].append(left)
    if mode=='shuffle':
        session_by_row={}
        # The donor stays in the original compatible session but at a random other position.
        from prism.core.sequence import build_sessions
        for session in build_sessions(np.arange(len(ids)),meta,session_gap_seconds,'anonymous_date'):
            for row in session:session_by_row[int(row)]=session
        original=peers;peers=[]
        for i,pp in enumerate(original):
            new=[]
            for j in pp:
                candidates=[int(k) for k in session_by_row[j] if k!=i and k!=j and k not in new]
                if not candidates:candidates=[int(k) for k in session_by_row[j] if k!=i and k not in new]
                if not candidates:raise ValueError('cannot preserve shuffled peer count')
                k=int(rng.choice(candidates));replacement_count+=int(k!=j);new.append(k)
            peers.append(new)
        assert [len(p) for p in original]==[len(p) for p in peers]
    return peers,meta,replacement_count

def features(bank,base,peers):
    sqrt=np.sqrt(bank);own=sqrt.reshape(len(bank),-1);agg=np.empty_like(own);votes=np.zeros((len(bank),40));counts=[]
    for i,pp in enumerate(peers):
        members=[i,*pp];agg[i]=sqrt[members].mean(0).ravel();votes[i]=np.bincount(base[members],minlength=40)/len(members);counts.append(len(pp))
    return np.c_[own,agg,votes,counts].astype(np.float32)

def apply_group(base,p,threshold):
    q=p.argmax(1);score=p[np.arange(len(p)),q]-p[np.arange(len(p)),base];out=base.copy();mask=(q!=base)&(score>=threshold['threshold']);out[mask]=q[mask];return out

def select_group(base,p,y):
    from prism.core.thresholds import choose_threshold
    if np.array_equal(base,p.argmax(1)):return {'threshold':2.,'net':0,'rescue':0,'harm':0,'changed':0}
    return choose_threshold(base,p.argmax(1),p,y)

def select_seq(base,p,q,y):
    from prism.core.sequence_thresholds import gate_features,select_gate
    if np.array_equal(base,q):return {'score_index':0,'threshold':2.,'net':0,'rescue':0,'harm':0,'changed':0}
    return select_gate(base,q,gate_features(p,base,q),y)

def apply_seq(base,p,q,gate):
    from prism.core.sequence_thresholds import gate_features
    accept=(q!=base)&(gate_features(p,base,q)[:,gate['score_index']]>=gate['threshold']);out=base.copy();out[accept]=q[accept];return out


def load_context(fold):
    with np.load(COMP/'runs'/RUNS[0]/f'fold{fold}_banks.npz', allow_pickle=False) as z:
        sid=z['inner_sample_ids'].astype(str);tid=z['outer_sample_ids'].astype(str)
        su=z['inner_users'].astype(str);tu=z['outer_users'].astype(str)
    si={s:i for i,s in enumerate(sid)};src=[];tgt=[];inner=None
    for run in RUNS:
        folder=COMP/'runs'/run
        if run not in (RUNS[1],RUNS[4]):
            with np.load(folder/f'fold{fold}_banks.npz',allow_pickle=False) as z:
                a={s:i for i,s in enumerate(z['inner_sample_ids'].astype(str))}
                b={s:i for i,s in enumerate(z['outer_sample_ids'].astype(str))}
                sx=z['inner_probability_bank'][[a[s] for s in sid]]
                tx=z['outer_probability_bank'][[b[s] for s in tid]]
        else:
            with np.load(folder/f'fold{fold}/outer/bank.npz',allow_pickle=False) as z:
                tx=z['probabilities']
            sx=np.empty((len(sid),tx.shape[1],40));part=np.full(len(sid),-1,int)
            for j in range(3):
                with np.load(folder/f'fold{fold}/inner{j}/bank.npz',allow_pickle=False) as z:
                    pos=np.array([si[s] for s in z['sample_ids'].astype(str)])
                    sx[pos]=z['probabilities'];part[pos]=j
            if inner is None:inner=part
        src.append(sx);tgt.append(tx)
    with np.load(COMP/'runs'/RUNS[1]/f'fold{fold}/outer/bank.npz',allow_pickle=False) as z:
        lookup=dict(zip(z['selection_ids'].astype(str),z['selection_labels']))
        y=np.array([lookup[s] for s in sid])
    return sid,tid,su,tu,np.concatenate(src,1),np.concatenate(tgt,1),y,inner


def online(p,model,blocks):
    covered={int(i) for chunk in blocks for i in chunk}
    chunks=blocks+[np.array([i]) for i in range(len(p)) if i not in covered]
    out=np.empty_like(p,dtype=np.float64)
    for chunk in chunks:
        alpha=model.start_log_probability.copy()
        for t,i in enumerate(chunk):
            if t:alpha=logsumexp(alpha[:,None]+model.bigram_log_probability,axis=0)
            alpha=alpha+np.log(np.maximum(p[i],1e-300))
            alpha-=logsumexp(alpha);out[i]=np.exp(alpha)
    return out.argmax(1)


def source_record(name,y,base,prediction):
    a=base==y;b=prediction==y
    rescue=int((~a & b).sum());harm=int((a & ~b).sum())
    return {'path':name,'net':rescue-harm,'rescues':rescue,'harms':harm,'changed':int((base!=prediction).sum())}


def construct(contexts,config,folder):
    started=time.perf_counter();ids=[];folds=[];joined=[];direct=[];selections=[]
    for f,c in enumerate(contexts):
        sid,tid,su,tu,sbank,tbank,y,inner=c
        cv=[(np.flatnonzero(inner!=j),np.flatnonzero(inner==j)) for j in range(3)]
        sx=np.sqrt(sbank).reshape(len(sid),-1);tx=np.sqrt(tbank).reshape(len(tid),-1)
        sd=crossfit(sx,y,cv);dm=stack_fit(sx,y);td=prob(dm,tx)
        sb=sd.argmax(1);tb=td.argmax(1)
        gap=config['session_gap_seconds']
        sp,sm,_=graph(sid,sb,sbank,METADATA,gap);tp,tm,_=graph(tid,tb,tbank,METADATA,gap)
        sfx=features(sbank,sb,sp);tfx=features(tbank,tb,tp)
        sg=crossfit(sfx,y,cv,True);gm=stack_fit(sfx,y,True);tg=prob(gm,tfx)
        gg=select_group(sb,sg,y);sclass=apply_group(sb,sg,gg);tclass=apply_group(tb,tg,gg)
        sblocks=build_sessions(np.arange(len(sid)),sm,gap,'anonymous_date')
        tblocks=build_sessions(np.arange(len(tid)),tm,gap,'anonymous_date')
        tr=fit_transition_model(y,sblocks,40,config['trigram_backoff'],alpha=.25)
        cfg=DecoderConfig(gap,config['transition_lambda'],config['trigram_backoff'],50)
        def decode(p,b,blocks):
            return decode_sessions(np.log(np.maximum(.65*p+.35*np.eye(40)[b],1e-9)),blocks,tr,cfg)
        sq=decode(sg,sclass,sblocks);tq=decode(tg,tclass,tblocks)
        qgate=select_seq(sclass,sg,sq,y)
        ss=apply_seq(sclass,sg,sq,qgate);ts=apply_seq(tclass,tg,tq,qgate)
        ds=decode(sd,sb,sblocks);dt=decode(td,tb,tblocks)
        dgate=select_seq(sb,sd,ds,y)
        dsg=apply_seq(sb,sd,ds,dgate);dtg=apply_seq(tb,td,dt,dgate)
        paths={'direct':sb,'group':sclass,'sequence':ss,'direct_sequence':dsg,'direct_sequence_nogate':ds,'online_forward':online(sd,tr,sblocks)}
        records=[source_record(name,y,sb,pp) for name,pp in paths.items()]
        winner=max(records,key=lambda row:(row['net'],row['rescues'],-row['harms'],-row['changed']))['path']
        target={'direct':tb,'group':tclass,'sequence':ts,'direct_sequence':dtg,'direct_sequence_nogate':dt,'online_forward':online(td,tr,tblocks)}
        joined.append(target[winner].astype(np.int64));direct.append(tb.astype(np.int64))
        ids.extend(tid);folds.extend([f]*len(tid))
        selections.append({'fold':f,'selected_path':winner,'group_gate':gg,'sequence_gate':qgate,'direct_sequence_gate':dgate})
        print(f'{folder.name}: fold {f}, path={winner}',flush=True)
    folder.mkdir()
    result={'sample_ids':np.array(ids),'outer_fold':np.array(folds),'prediction':np.concatenate(joined),'direct':np.concatenate(direct)}
    np.savez_compressed(folder/'targets.npz',**result)
    dump(folder/'selection.json',{'configuration':config,'folds':selections})
    return result,selections,time.perf_counter()-started


def reproduce(result,selections):
    expected_paths=['sequence','direct_sequence','direct_sequence']
    actual=[row['selected_path'] for row in selections]
    if actual!=expected_paths:
        raise SystemExit(f'复现未通过：各折路径={actual}，预期={expected_paths}；已停止。')
    with np.load(REFERENCE/'frozen_predictions.npz',allow_pickle=False) as z:
        lookup={s:i for i,s in enumerate(z['evaluation_ids'].astype(str))}
        pos=np.array([lookup[s] for s in result['sample_ids']])
        expected_direct=z['evaluation_prediction__direct'][pos]
        expected_selected=np.empty(len(pos),int)
        for f,path in enumerate(expected_paths):
            mask=result['outer_fold']==f
            expected_selected[mask]=z['evaluation_prediction__'+path][pos[mask]]
    nd=int((result['direct']!=expected_direct).sum());npred=int((result['prediction']!=expected_selected).sum())
    if len(pos)!=2470 or nd or npred:
        raise SystemExit(f'复现未通过：片段数={len(pos)}，Direct 差异={nd}，PRISM 差异={npred}；已停止。')
    print('复现通过：2,470 个片段的 Direct / PRISM 与论文冻结预测逐片段一致（74.97 / 79.95），路径 D→G→S / D→S / D→S。',flush=True)


def evaluate(configurations,elapsed):
    frozen=[]
    for name,key,value,config in configurations:
        folder=BASE/'runs_cuhkx'/name
        with np.load(folder/'targets.npz',allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        frozen.append((name,key,value,config,arrays,json.loads((folder/'selection.json').read_text(encoding='utf-8'))['folds']))
    print('12 个配置的预测、路径和阈值已全部写盘；现在读取评估标签。',flush=True)
    with np.load(TRUTH,allow_pickle=False) as z:
        lookup=dict(zip(z['sample_ids'].astype(str),z['labels'].astype(int)))
    final=next(row for row in frozen if row[3]==FINAL)
    final_arrays=final[4];y=np.array([lookup[s] for s in final_arrays['sample_ids']])
    direct_f1=100*metrics(y,final_arrays['direct'],40)['macro_f1_fixed40']
    prism_f1=100*metrics(y,final_arrays['prediction'],40)['macro_f1_fixed40']
    if f'{direct_f1:.2f}'!='74.97' or f'{prism_f1:.2f}'!='79.95':
        raise SystemExit(f'复现得分未通过：Direct={direct_f1:.4f}，PRISM={prism_f1:.4f}；已停止。')
    rows=[]
    for name,key,value,config,a,selection in frozen:
        yy=np.array([lookup[s] for s in a['sample_ids']])
        f1=100*metrics(yy,a['prediction'],40)['macro_f1_fixed40']
        row={'parameter':key,'value':value,**config,'macro_f1_fixed40':f1,'delta_vs_direct':f1-direct_f1}
        for f in range(3):
            mask=a['outer_fold']==f
            row[f'fold{f}_macro_f1']=100*metrics(yy[mask],a['prediction'][mask],40)['macro_f1_fixed40']
            row[f'fold{f}_path']=PATH_NAMES[selection[f]['selected_path']]
        row['changed_vs_final']=int((a['prediction']!=final_arrays['prediction']).sum())
        rows.append(row)
    with (BASE/'results_cuhkx.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# CUHK-X 解码超参数敏感性','',f'三折合并 2,470 个片段；固定 40 类 macro-F1，缺席类计 0；分数单位为 %，Δ 单位为百分点。Direct = {direct_f1:.4f}（74.97），最终配置 PRISM = {prism_f1:.4f}（79.95）。* 为最终配置，在三个参数范围中共享且只运行一次。','',
           '| 常数 | 取值 | 合并 macro-F1 | Δ vs Direct | fold 0 F1 | fold 1 F1 | fold 2 F1 | fold 0 路径 | fold 1 路径 | fold 2 路径 | 标签变化数 vs 最终 |',
           '|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|']
    for r in rows:
        star='*' if all(r[k]==v for k,v in FINAL.items()) else ''
        lines.append(f"| `{r['parameter']}` | {r['value']:g}{star} | {r['macro_f1_fixed40']:.4f} | {r['delta_vs_direct']:+.4f} | {r['fold0_macro_f1']:.4f} | {r['fold1_macro_f1']:.4f} | {r['fold2_macro_f1']:.4f} | {r['fold0_path']} | {r['fold1_path']} | {r['fold2_path']} | {r['changed_vs_final']} |")
    lines+=['','## 变化范围','']
    for key in GRID:
        members=[row for row in frozen if row[1]==key or row[3]==FINAL]
        scores=[100*metrics(y,row[4]['prediction'],40)['macro_f1_fixed40'] for row in members]
        lines.append(f'- `{key}`：{min(scores):.4f}–{max(scores):.4f}，最大值减最小值 = {max(scores)-min(scores):.4f} 个百分点。')
        if all(np.array_equal(members[0][4]['prediction'],row[4]['prediction']) for row in members[1:]):
            lines.append(f'  `{key}`：该键可能未生效。')
    lines+=['',f'专家数据实际路径：`{COMP.as_posix()}/runs/` 下 P427–P433 的 23 专家 outer/inner bank；源标签为 P428 各折 outer bank 的 `selection_labels`。',
            f'元信息实际路径：`{METADATA.as_posix()}`；评估标签实际路径：`{TRUTH.as_posix()}`。',
            '实际使用的键名：`transition_lambda` → `DecoderConfig.transition_weight`；`trigram_backoff` → `fit_transition_model(..., trigram_backoff)`；`session_gap_seconds` → 重复同伴和解码两处的 `build_sessions(..., gap_seconds)`。',
            f'总耗时：{elapsed/60:.2f} 分钟（CPU；仅重拟合 sklearn stacker/group，未训练或推理神经网络）。']
    (BASE/'RESULTS_cuhkx.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'完成：Direct={direct_f1:.4f}，最终 PRISM={prism_f1:.4f}；结果已写入 RESULTS_cuhkx.md 和 results_cuhkx.csv。',flush=True)


def main():
    global BASE, COMP, REFERENCE, TRUTH, METADATA
    parser=argparse.ArgumentParser()
    parser.add_argument('--comp',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--truth',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--transition-weight',type=float,default=.45)
    parser.add_argument('--trigram-backoff',type=float,default=1.)
    parser.add_argument('--session-gap-seconds',type=float,default=30.)
    args=parser.parse_args()
    BASE=args.output
    COMP=args.comp
    REFERENCE=args.reference
    TRUTH=args.truth
    METADATA=args.metadata
    if (args.transition_weight,args.trigram_backoff,args.session_gap_seconds)!=(.45,1.,30.):
        raise SystemExit('本入口按固定单因素网格运行，复现参数必须为 0.45 / 1.0 / 30。')
    started=time.perf_counter();root=BASE/'runs_cuhkx';root.mkdir()
    configurations=[]
    for key,values in GRID.items():
        for value in values:
            if key!='transition_lambda' and value==FINAL[key]:continue
            config={**FINAL,key:value};name=f'{key}_{value:g}'
            configurations.append((name,key,value,config))
    with threadpool_limits(2):
        baseline_started=time.perf_counter()
        contexts=[load_context(f) for f in range(3)]
        baseline=next(row for row in configurations if row[3]==FINAL)
        result,selections,_=construct(contexts,FINAL,root/baseline[0])
        reproduce(result,selections)
        baseline_seconds=time.perf_counter()-baseline_started
        print(f'预计总耗时约 {baseline_seconds*12/60:.1f} 分钟',flush=True)
        for name,key,value,config in configurations:
            if config==FINAL:continue
            construct(contexts,config,root/name)
        evaluate(configurations,time.perf_counter()-started)


if __name__=='__main__':main()
