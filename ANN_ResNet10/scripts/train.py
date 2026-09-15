"""Train/evaluate ResNet-10 on FashionMNIST."""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models import ResNet10

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def loaders(args):
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train=datasets.FashionMNIST(args.data, train=True, download=True, transform=transform)
    test=datasets.FashionMNIST(args.data, train=False, download=True, transform=transform)
    if args.train_size: train=Subset(train, range(min(args.train_size,len(train))))
    if args.test_size: test=Subset(test, range(min(args.test_size,len(test))))
    return DataLoader(train,args.batch_size,shuffle=True,num_workers=0), DataLoader(test,args.batch_size,shuffle=False,num_workers=0)

def evaluate(model, loader, device):
    model.eval(); correct=total=loss_sum=0; criterion=nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y in loader:
            out=model(x.to(device)); y=y.to(device); loss_sum += criterion(out,y).item()*y.size(0); correct += (out.argmax(1)==y).sum().item(); total += y.size(0)
    return {"loss":loss_sum/total,"acc":correct/total,"samples":total}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=Path('data')); p.add_argument('--output',type=Path,default=Path('checkpoints/resnet10_fashionmnist.pt')); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--train-size',type=int); p.add_argument('--test-size',type=int); p.add_argument('--seed',type=int,default=20260911); p.add_argument('--eval-only',action='store_true'); args=p.parse_args(); seed_all(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); train_loader,test_loader=loaders(args); model=ResNet10().to(device); criterion=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); history=[]
    if args.eval_only:
        ckpt=torch.load(args.output,map_location=device); model.load_state_dict(ckpt['model']); metrics=evaluate(model,test_loader,device); print(json.dumps(metrics)); return 0
    for epoch in range(1,args.epochs+1):
        model.train(); seen=correct=loss_sum=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); out=model(x); loss=criterion(out,y); loss.backward(); opt.step(); loss_sum += loss.item()*y.size(0); correct += (out.argmax(1)==y).sum().item(); seen += y.size(0)
        metrics=evaluate(model,test_loader,device); row={"epoch":epoch,"train_loss":loss_sum/seen,"train_acc":correct/seen,"test":metrics}; history.append(row); print(json.dumps(row),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'config':vars(args),'history':history},args.output); args.output.with_suffix('.json').write_text(json.dumps({'device':str(device),'history':history},indent=2),encoding='utf8'); print(json.dumps({'checkpoint':str(args.output),'final':history[-1]})); return 0
if __name__=='__main__': raise SystemExit(main())
