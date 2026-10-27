"""Within-workout pseudo-bout peers and relative-phase matching."""
from __future__ import annotations
import numpy as np

def bout_repeat_peers(workouts,times,features,probabilities,min_gap=10.,max_gap=600.,threshold=.85,max_bouts=8):
    """Label-free repeat sets across pseudo-bouts, aligned by relative phase.

    Re-exported by ``prism.peers`` for MM-Fit pseudo-bout peer construction.
    Sustained same-class windows define candidate bouts only; uncertain tiny
    bouts remain valid. Self-bout and temporally overlapping candidates are
    excluded. There is no unique-class constraint or maximum group size three.
    """
    workouts=np.asarray(workouts);times=np.asarray(times);x=np.asarray(features,np.float32)
    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8);pred=probabilities.argmax(1)
    peers=[[] for _ in times];bout_count=0;edge_count=0
    for workout in np.unique(workouts):
        order=np.flatnonzero(workouts==workout);order=order[np.argsort(times[order],kind='stable')]
        cut=np.r_[0,np.flatnonzero((np.diff(pred[order])!=0)|(np.diff(times[order])>1.5))+1,len(order)]
        bouts=[order[a:b] for a,b in zip(cut[:-1],cut[1:])];bout_count+=len(bouts)
        desc=np.stack([x[b].mean(0) for b in bouts]);desc/=np.maximum(np.linalg.norm(desc,axis=1,keepdims=True),1e-8)
        sim=desc@desc.T
        for j,rows in enumerate(bouts):
            candidates=[]
            for k,other in enumerate(bouts):
                gap=max(times[other[0]]-times[rows[-1]],times[rows[0]]-times[other[-1]])
                if j!=k and min_gap<=gap<=max_gap and sim[j,k]>=threshold:candidates.append(k)
            selected=sorted(candidates,key=lambda k:(-sim[j,k],k))[:max_bouts]
            for n,row in enumerate(rows):
                phase=n/max(len(rows)-1,1)
                peers[row]=[int(bouts[k][round(phase*(len(bouts[k])-1))]) for k in selected]
                edge_count+=len(peers[row])
    return tuple(tuple(p) for p in peers),{'pseudo_bouts':bout_count,'directed_phase_edges':edge_count,
        'rows_with_peers':sum(bool(p) for p in peers),'test_labels_received':False,'label_free_segmentation':True}
