"""Independent test slice for selected shared-MR sweep checkpoints."""
import argparse,json,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from IMC_ResNet.models.finetuned_snn import load_finetuned,fuse_explicit_hidden,make_explicit_quantized_reference
from SNN_ResNet10.models.snn_resnet10 import reset


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--size',type=int,default=1024);p.add_argument('--offset',type=int,default=128);a=p.parse_args()
    torch.set_num_threads(2)
    data=datasets.FashionMNIST(ROOT/'SNN_ResNet10/data',train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.2860,),(.3530,))]))
    loader=DataLoader(Subset(data,range(a.offset,a.offset+a.size)),32)
    model,_=load_finetuned(a.checkpoint,ROOT/'SNN_ResNet10/checkpoints/calibrated.json');model=make_explicit_quantized_reference(fuse_explicit_hidden(model))
    correct={16:0,64:0};predictions={16:[],64:[]};labels=[]
    with torch.no_grad():
        for i,(x,y) in enumerate(loader):
            reset(model);total=0;labels+=y.tolist()
            for t in range(1,65):
                total=total+model(x.unsqueeze(0))[0]
                if t in correct:
                    pred=total.argmax(1);correct[t]+=int((pred==y).sum());predictions[t]+=pred.tolist()
            if i%8==0: print(f'{i+1}/{len(loader)} batches',flush=True)
    report=dict(checkpoint=str(a.checkpoint),test_range=[a.offset,a.offset+a.size],samples=a.size,correct=correct,accuracy={k:v/a.size for k,v in correct.items()},predictions=predictions,labels=labels)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf8');print(report['accuracy'],flush=True)


if __name__=='__main__':main()
