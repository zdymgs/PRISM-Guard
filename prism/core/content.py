"""Content stacking and own-plus-peer expert features."""
from __future__ import annotations
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .thresholds import gate_features

def stack_fit(x,y,weighted=False):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.03,solver='lbfgs',max_iter=1000,random_state=20260912))
    model.fit(x,y,logisticregression__sample_weight=np.where(x[:,-1]>0,2.,1.) if weighted else np.ones(len(x)))
    return model


def prob(model,x,classes=40):
    p=np.zeros((len(x),classes));p[:,model[-1].classes_.astype(int)]=model.predict_proba(x);return p


def crossfit(x,y,cv,weighted=False,classes=40):
    p=np.zeros((len(x),classes))
    for fit,held in cv:p[held]=prob(stack_fit(x[fit],y[fit],weighted),x[held],classes=classes)
    return p


def features(bank,base,peers,classes=40):
    sqrt=np.sqrt(bank);own=sqrt.reshape(len(bank),-1);agg=np.empty_like(own);votes=np.zeros((len(bank),classes));counts=[]
    for i,pp in enumerate(peers):
        members=[i,*pp];agg[i]=sqrt[members].mean(0).ravel();votes[i]=np.bincount(base[members],minlength=classes)/len(members);counts.append(len(pp))
    return np.c_[own,agg,votes,counts].astype(np.float32)


def mean_prob(bank,peers):
    p=np.stack([np.sqrt(bank[[i,*pp]]).mean(axis=(0,1)) for i,pp in enumerate(peers)]);return p/p.sum(1,keepdims=True)


def apply_group(base,p,threshold):
    q=p.argmax(1);score=p[np.arange(len(p)),q]-p[np.arange(len(p)),base];out=base.copy();mask=(q!=base)&(score>=threshold['threshold']);out[mask]=q[mask];return out


def apply_seq(base,p,q,gate):
    from .thresholds import gate_features
    accept=(q!=base)&(gate_features(p,base,q)[:,gate['score_index']]>=gate['threshold']);out=base.copy();out[accept]=q[accept];return out
