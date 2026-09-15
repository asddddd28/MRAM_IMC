"""Controlled shared-MR QAT ablations with validation separated from test."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from SNN_ResNet10.models.snn_resnet10 import reset
from SNN_ResNet10.models.mr_qat import QATConv2d,MRProxy
from SNN_ResNet10.scripts.finetune_mr_qat import export,evaluate
from IMC_ResNet.models.finetuned_snn import load_finetuned,fuse_explicit_hidden,explicit_targets

CONFIGS=[
 dict(name='mt_control',kd=0.,rho=0.,stem=False,rate=0.),
 dict(name='mt_mr_low',kd=0.,rho=.1,stem=False,rate=0.),
 dict(name='mt_mr_high',kd=0.,rho=.5,stem=False,rate=0.),
 dict(name='kd_control',kd=1.,rho=0.,stem=False,rate=0.),
 dict(name='kd_mr',kd=1.,rho=.3,stem=False,rate=0.),
 dict(name='stem_kd_control',kd=1.,rho=0.,stem=True,rate=0.),
 dict(name='stem_kd_mr',kd=1.,rho=.3,stem=True,rate=0.),
 dict(name='stem_kd_mr_rate',kd=1.,rho=.3,stem=True,rate=.2),
]


def task_loss(output,labels,teacher=None,kd=0.):
    logits=[output[:16].mean(0),output.mean(0)]
    loss=sum(F.cross_entropy(z,labels) for z in logits)/2
    if teacher is not None and kd:
        targets=[teacher[:16].mean(0),teacher.mean(0)]
        loss=loss+kd*sum(F.kl_div(F.log_softmax(z/2,dim=-1),F.softmax(y/2,dim=-1),reduction='batchmean')*4 for z,y in zip(logits,targets))/2
    return loss


def rate_loss(rates,teacher_rates):
    # Per-channel spatial/batch average, four successive 16-step windows.
    student=torch.stack(rates).reshape(4,16,-1).mean(1)
    budget=(torch.stack(teacher_rates).mean(0)*.85).clamp_min(.02)
    return F.relu(student-budget).square().mean()


def diagnose(model,loader,independent=False):
    proxy=MRProxy(model.layer1.conv1,target=29 if independent else 248,positions=0,shared_bits=None if independent else 9,independent_cancel=independent)
    handles=[model.layer1.conv1.register_forward_hook(proxy.observe),model.layer1.lif1.register_forward_hook(proxy.fire)]
    lo=hi=outside=0
    maximum=outside5=0
    try:
        with torch.no_grad():
            for x,_ in loader:
                reset(model);proxy.reset()
                model(x.unsqueeze(0).expand(64,-1,-1,-1,-1))
                maximum=max(maximum,proxy.max_mr);outside5+=proxy.outside_5bit
                lo=min(lo,proxy.shared_min);hi=max(hi,proxy.shared_max);outside+=proxy.shared_outside
    finally:
        for h in handles:h.remove()
        reset(model);proxy.reset()
    if independent:
        return dict(max_mr=maximum,outside_5bit=outside5,scope="first layer full spatial; 8 diagnostic test samples")
    return dict(minimum=lo,maximum=hi,outside_9bit=outside,scope='first layer full spatial; diagnostic test indices 0,1,2,3,4,5,64,115')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--independent-cancel',action='store_true')
    p.add_argument('--cases',nargs='+',default=[c['name'] for c in CONFIGS])
    p.add_argument('--init-checkpoint',type=Path,default=ROOT/'SNN_ResNet10/checkpoints/hard_reset_finetuned.pt')
    p.add_argument('--rho-multiplier',type=float,default=1.)
    p.add_argument('--train-size',type=int,default=64)
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--lr',type=float,default=.0005)
    p.add_argument('--output-dir',type=Path,default=ROOT/'SNN_ResNet10/checkpoints/shared_mr_sweep_v2')
    a=p.parse_args();torch.set_num_threads(4)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.2860,),(.3530,))])
    train=datasets.FashionMNIST(ROOT/'SNN_ResNet10/data',train=True,transform=tf)
    test=datasets.FashionMNIST(ROOT/'SNN_ResNet10/data',train=False,transform=tf)
    val=DataLoader(Subset(train,range(59000,59256)),32)
    risk=DataLoader(Subset(test,[0,1,2,3,4,5,64,115]),2)
    source,_=load_finetuned(a.init_checkpoint,ROOT/'SNN_ResNet10/checkpoints/calibrated.json')
    base=fuse_explicit_hidden(source)
    for name,_,_ in explicit_targets(base):
        conv=base.get_submodule(name)
        qat=QATConv2d(conv.in_channels,conv.out_channels,conv.kernel_size,conv.stride,conv.padding,bias=conv.bias is not None)
        qat.load_state_dict(conv.state_dict());base.set_submodule(name,qat)
    for node in base.modules():
        if isinstance(node,neuron.IFNode): node.detach_reset=True
    base.eval()
    teacher=copy.deepcopy(base)
    for par in teacher.parameters():par.requires_grad_(False)
    report=dict(config=vars(a).copy(),validation_indices=[59000,59256],baseline_accuracy=evaluate(base,val),baseline_mr=diagnose(base,risk,a.independent_cancel),cases=[],status='running')
    report['config']['output_dir']=str(a.output_dir)
    report['config']['init_checkpoint']=str(a.init_checkpoint)
    def save():
        tmp=a.output_dir/'summary.tmp'
        tmp.write_text(json.dumps(report,indent=2),encoding='utf8');tmp.replace(a.output_dir/'summary.json')
    save();print('BASELINE '+json.dumps({k:report[k] for k in ['baseline_accuracy','baseline_mr']}),flush=True)
    for config in CONFIGS:
        if config['name'] not in a.cases:continue
        config=dict(config)
        config['rho']*=a.rho_multiplier
        torch.manual_seed(20260914)
        model=copy.deepcopy(base).eval()
        for name,par in model.named_parameters():
            par.requires_grad_(name=='layer1.conv1.weight' or (config['stem'] and name.startswith('stem.0.')))
        params=[p for p in model.parameters() if p.requires_grad]
        proxy=MRProxy(model.layer1.conv1,target=29 if a.independent_cancel else 248,positions=64,shared_bits=None if a.independent_cancel else 9,independent_cancel=a.independent_cancel)
        handles=[model.layer1.conv1.register_forward_hook(proxy.observe),model.layer1.lif1.register_forward_hook(lambda m,i,s:proxy.fire(m,i,s.detach()))]
        rates=[];teacher_rates=[]
        if config['rate']:
            handles.append(model.stem.register_forward_hook(lambda m,i,s:rates.append(s.mean((0,2,3)))))
            th=teacher.stem.register_forward_hook(lambda m,i,s:teacher_rates.append(s.mean((0,2,3))))
        opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=0)
        loader=DataLoader(Subset(train,range(a.train_size)),2,shuffle=True,generator=torch.Generator().manual_seed(20260914))
        history=[];start=time.perf_counter()
        for epoch in range(a.epochs):
            for batch,(x,y) in enumerate(loader):
                reset(model);proxy.reset();reset(teacher);rates.clear();teacher_rates.clear();opt.zero_grad(set_to_none=True)
                sequence=x.unsqueeze(0).expand(64,-1,-1,-1,-1)
                with torch.no_grad():teach=teacher(sequence) if config['kd'] or config['rate'] else None
                output=model(sequence)
                task=task_loss(output,y,teach,config['kd'])
                terms=proxy.losses()
                mr=terms['overflow']+terms['peak']+.1*terms['quiet']
                rate=rate_loss(rates,teacher_rates) if config['rate'] else task*0
                regularizer=mr+config['rate']*rate
                factor=0.
                if config['rho']:
                    tg=torch.autograd.grad(task,params,retain_graph=True)
                    rg=torch.autograd.grad(regularizer,params,retain_graph=True)
                    tn=torch.sqrt(sum(g.square().sum() for g in tg));rn=torch.sqrt(sum(g.square().sum() for g in rg))
                    factor=min(1e5,config['rho']*float(tn)/max(float(rn),1e-12))
                loss=task+factor*regularizer
                loss.backward()
                assert torch.isfinite(loss) and all(torch.isfinite(par.grad).all() for par in params)
                torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
                history.append(dict(epoch=epoch,batch=batch,task=float(task.detach()),mr=float(mr.detach()),rate=float(rate.detach()),factor=factor,**proxy.report()))
                if batch%8==0 or batch+1==len(loader):print(f"{config['name']} epoch={epoch+1} batch={batch+1}/{len(loader)} task={float(task.detach()):.4f} MRmax={proxy.max_mr} C=[{proxy.shared_min},{proxy.shared_max}] seconds={time.perf_counter()-start:.1f}",flush=True)
        for h in handles:h.remove()
        if config['rate']:th.remove()
        proxy.reset();rates.clear();teacher_rates.clear();reset(teacher)
        accuracy=evaluate(model,val);diagnostic=diagnose(model,risk,a.independent_cancel)
        path=a.output_dir/(config['name']+'.pt')
        args=SimpleNamespace(leakage=1.,steps=64,target=29 if a.independent_cancel else 248,positions=64,beta=1.,gamma=.1,shared_bits=None if a.independent_cancel else 9,detach_reset=True,balance_mr_gradient=True)
        export(model,args,config['rho'],history,path)
        # Preserve exact experiment recipe and trainable scope with the checkpoint.
        ck=torch.load(path,map_location='cpu',weights_only=False);ck['experiment']={**config,'independent_cancel':a.independent_cancel};ck['trainable_parameters']=[n for n,p in model.named_parameters() if p.requires_grad];torch.save(ck,path)
        row=dict(**config,accuracy=accuracy,mr=diagnostic,checkpoint=str(path.resolve()),seconds=time.perf_counter()-start,history=history)
        report['cases'].append(row);save()
        print('RESULT '+json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
    report['status']='complete';save()


if __name__=='__main__':main()
