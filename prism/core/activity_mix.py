"""MMAct expert mixing and monotone posterior matching."""
from __future__ import annotations
import numpy as np

CLASSES=37; MATCH_THRESHOLD=.55; SKIP_PENALTY=.20; TRANSITION_WEIGHT=.45


def _prob(bank,mask,name):
 bank=np.asarray(bank,np.float64);mask=np.asarray(mask,bool)
 if bank.ndim!=3 or bank.shape[2]!=CLASSES or mask.shape!=bank.shape[:2]:raise ValueError(f"{name} bank shape mismatch")
 if not np.isfinite(bank).all() or np.any(bank<0):raise ValueError(f"{name} has invalid probabilities")
 sums=bank.sum(2)
 if np.any(mask&~np.isclose(sums,1,atol=1e-5)) or np.any((~mask)&(np.abs(bank).sum(2)>1e-10)):raise ValueError(f"{name} availability/probability mismatch")
 return bank,mask


def weighted_available_mix(bank,mask,expert_names,selection):
 bank,mask=_prob(bank,mask,"expert");names=list(map(str,expert_names));wanted=list(selection["names"]);weights=np.asarray(selection["weights"],np.float64)
 if len(wanted)!=len(weights) or not np.isclose(weights.sum(),1):raise ValueError("invalid fixed selection weights")
 try:idx=np.asarray([names.index(x) for x in wanted])
 except ValueError as e:raise ValueError("selected expert absent from bank") from e
 active=mask[:,idx]*weights[None];den=active.sum(1)
 if np.any(den<=0):raise ValueError("row has no selected direct expert")
 out=(bank[:,idx]*active[:,:,None]).sum(1)/den[:,None];return out/out.sum(1,keepdims=True)


def _monotone(sim,left,right):
 matrix=sim[np.ix_(left,right)];m,n=matrix.shape;score=np.full((m+1,n+1),-np.inf);action=np.zeros((m+1,n+1),np.int8);score[0,0]=0;score[1:,0]=-SKIP_PENALTY*np.arange(1,m+1);score[0,1:]=-SKIP_PENALTY*np.arange(1,n+1);action[1:,0]=1;action[0,1:]=2
 for i in range(1,m+1):
  for j in range(1,n+1):
   opts=(score[i-1,j-1]+matrix[i-1,j-1]-MATCH_THRESHOLD,score[i-1,j]-SKIP_PENALTY,score[i,j-1]-SKIP_PENALTY);action[i,j]=np.argmax(opts);score[i,j]=opts[action[i,j]]
 found=[];i=m;j=n
 while i or j:
  move=int(action[i,j])
  if move==0:
   if matrix[i-1,j-1]>=MATCH_THRESHOLD:found.append((int(left[i-1]),int(right[j-1]),float(matrix[i-1,j-1])))
   i-=1;j-=1
  elif move==1:i-=1
  else:j-=1
 return tuple(reversed(found))
