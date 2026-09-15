"""纯整数 SimpleCluster：面向推理、调试和 RTL 对照。"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import numpy as np
from numbers import Integral
BETA=(1,2,4,8,-16)

@dataclass(frozen=True)
class SimpleConfig:
    scu_bits:int=4; mr_bits:int=8; theta:int=560; overflow_policy:str="error"
    def __post_init__(self):
        if self.scu_bits not in (3,4) or self.mr_bits<1 or type(self.theta) is not int: raise ValueError("invalid SimpleConfig")
        if self.overflow_policy not in {"error","wide_reference","wrap","saturate"}: raise ValueError("invalid overflow_policy")
    @property
    def L(self): return 1 << self.scu_bits
    @property
    def mr_max(self): return (1 << self.mr_bits)-1

@dataclass
class SimpleState:
    scu: np.ndarray
    mr: np.ndarray
    @classmethod
    def zeros(cls): return cls(np.zeros((16,5),dtype=object),np.zeros((16,5),dtype=object))
    def copy(self): return SimpleState(self.scu.copy(),self.mr.copy())

@dataclass
class SimpleResult:
    spike: bool
    state: SimpleState
    candidate: SimpleState
    trace: dict

class SimpleModelError(RuntimeError):
    def __init__(self,code,**info): self.code=code; self.info={"code":code,**info}; super().__init__(f"{code}: {info}")

def _ints(value,shape,name,low=None,high=None):
    a=np.asarray(value,dtype=object)
    if a.shape != shape: raise ValueError(f"{name} shape must be {shape}")
    for v in a.flat:
        if not isinstance(v,Integral) or (low is not None and v<low) or (high is not None and v>high): raise ValueError(f"invalid {name}")
    return np.array([int(v) for v in a.flat],dtype=object).reshape(shape)

def encode_weights(weights):
    w=np.asarray(weights,dtype=object)
    if w.shape!=(16,36): raise ValueError("weights shape must be (16,36)")
    for v in w.flat:
        if not isinstance(v,Integral) or not -16<=v<=15: raise ValueError("weights must be int5")
    return ((w.astype(np.int64)[...,None] & 31) >> np.arange(5)) & 1

def _pmac(inputs,bits):
    x=_ints(inputs,(16,36),"inputs",0,1); return np.array([[sum(int(x[m,i])*int(bits[m,i,b]) for i in range(36)) for b in range(5)] for m in range(16)],dtype=object)

def _reduce(state):
    local_s=np.array([sum(int(state.scu[m,b])*BETA[b] for b in range(5)) for m in range(16)],dtype=object)
    local_g=np.array([sum(int(state.mr[m,b])*BETA[b] for b in range(5)) for m in range(16)],dtype=object)
    return {"local_s":local_s,"local_g":local_g,"local_u":local_s+16*local_g,"total_s":sum(local_s),"total_g":sum(local_g),"total_u":sum(local_s)+16*sum(local_g)}

class SimpleCluster:
    """短小的单 Cluster 整数执行器；每步只有一次比较和一次提交。"""
    def __init__(self,config=None): self.config=config or SimpleConfig(); self.state=SimpleState.zeros(); self.logical_step=0
    def reset(self): self.state=SimpleState.zeros(); self.logical_step=0
    def snapshot(self): return {"state":self.state.copy(),"logical_step":self.logical_step}
    def restore(self,snapshot): self.state=snapshot["state"].copy(); self.logical_step=snapshot["logical_step"]
    def step(self,inputs,weights,*,theta=None,commit=True): return self.step_tiles([(inputs,weights)],theta=theta,commit=commit)
    def step_tiles(self,tiles,*,theta=None,commit=True):
        cfg=self.config; candidate=self.state.copy(); folds=[]; total=np.zeros((16,5),dtype=object)
        try:
            for fold,(inputs,weights) in enumerate(tiles):
                bits=weights if np.asarray(weights).shape==(16,36,5) else encode_weights(weights)
                counts=_pmac(inputs,bits); carry=np.empty((16,5),dtype=object)
                for m in range(16):
                    for b in range(5):
                        z=int(candidate.scu[m,b])+int(counts[m,b]); carry[m,b],candidate.scu[m,b]=divmod(z,cfg.L); candidate.mr[m,b]=int(candidate.mr[m,b])+int(carry[m,b])
                        if candidate.mr[m,b]>cfg.mr_max:
                            if cfg.overflow_policy=="error": raise SimpleModelError("mr_overflow",fold=fold,macro=m,bit=b,value=int(candidate.mr[m,b]))
                            if cfg.overflow_policy=="wrap": candidate.mr[m,b]%=cfg.mr_max+1
                            if cfg.overflow_policy=="saturate": candidate.mr[m,b]=cfg.mr_max
                total += counts; folds.append({"fold":fold,"counts":counts.copy(),"carry":carry.copy(),"candidate":candidate.copy()})
            if not folds: raise ValueError("step requires at least one tile")
            reduced=_reduce(candidate); threshold=cfg.theta if theta is None else int(theta); spike=reduced["total_u"]>=threshold; committed=SimpleState.zeros() if spike else candidate.copy()
            trace={"step":self.logical_step,"folds":folds,"total_counts":total,"theta":threshold,"spike":bool(spike),**reduced}
            result=SimpleResult(bool(spike),committed.copy(),candidate.copy(),trace)
        except Exception:
            raise
        if commit: self.state=committed.copy(); self.logical_step+=1
        return result
    def tdp_waveform(self):
        from .tdp import scu_to_tdp
        return scu_to_tdp(self.state.scu,bits=5)
