"""Evaluate one signed shared MR per Macro across every mapped convolution."""
import argparse
import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from IMC_ResNet.models.cluster_backend import ClusterConv2d
from IMC_ResNet.models.shared_macro_mr import SharedMacroMR, guaranteed_bound, signed_bits
from IMC_ResNet.models.finetuned_snn import load_finetuned, fuse_explicit_hidden, make_explicit_quantized_reference, make_explicit_cluster, SingleStepExplicit
from IMC_ResNet.scripts.run_cluster_inference import evaluate


def main():
    torch.set_num_threads(4)
    snn=ROOT/'SNN_ResNet10'
    parser=argparse.ArgumentParser()
    parser.add_argument('--independent-cancel',action='store_true')
    parser.add_argument('--checkpoint',type=Path,default=snn/'checkpoints/hard_reset_finetuned.pt')
    parser.add_argument('--output',type=Path,default=ROOT/'IMC_ResNet/checkpoints/shared_macro_mr.json')
    parser.add_argument('--indices',nargs='+',type=int)
    parser.add_argument('--test-size',type=int)
    parser.add_argument('--test-offset',type=int,default=128)
    args=parser.parse_args()
    indices=args.indices or (list(range(args.test_offset,args.test_offset+args.test_size)) if args.test_size else [0,1,2,3,4,5,64,115])
    data=datasets.FashionMNIST(snn/'data',train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.2860,),(.3530,))]))
    loader=DataLoader(Subset(data,indices),2)
    original,meta=load_finetuned(args.checkpoint,snn/'checkpoints/calibrated.json')
    fused=fuse_explicit_hidden(original)
    reference=make_explicit_quantized_reference(fused)
    actual,mapping=make_explicit_cluster(fused)
    from IMC_ResNet.models.independent_cancel_mr import IndependentCancelMR
    for module in actual.modules():
        if isinstance(module,ClusterConv2d): module.bank=IndependentCancelMR() if args.independent_cancel else SharedMacroMR()
    control=evaluate(SingleStepExplicit(reference),loader,[16,64],'int5_digital')
    print('Digital reference complete',flush=True)
    result=evaluate(SingleStepExplicit(actual),loader,[16,64],'independent_cancel_mr' if args.independent_cancel else 'shared_macro_mr',cluster=True)
    layers={}
    for name,module in actual.named_modules():
        if isinstance(module,ClusterConv2d):
            layers[name]={**module.bank.report(),'fan_in':module.fan_in,'folds':module.folds,'generic_T64_bound':guaranteed_bound(64,module.folds)}
    if args.independent_cancel:
        disagreements={t:sum(a!=b for a,b in zip(pred,control['predictions_by_steps'][t])) for t,pred in result['predictions_by_steps'].items()}
        assert all(n==0 for n in disagreements.values())
        report=dict(status='complete',indices=indices,checkpoint=str(args.checkpoint),scope='independent MR; original mapping; all 11 convolutions; T16/T64; post-cancel before reset',digital=control,independent=result,disagreements=disagreements,layers=layers,global_max_mr=max(v['max_mr'] for v in layers.values()))
        args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
        print(json.dumps({n:{k:v[k] for k in ['max_mr','plane_max','max_local_integer_error']} for n,v in layers.items()}),flush=True)
        return
    lo=min(v['minimum'] for v in layers.values()); hi=max(v['maximum'] for v in layers.values())
    disagreements={t:sum(a!=b for a,b in zip(pred,control['predictions_by_steps'][t])) for t,pred in result['predictions_by_steps'].items()}
    assert all(n==0 for n in disagreements.values())
    report=dict(status='complete',indices=indices,steps=[16,64],checkpoint=meta,mapping=mapping,
        scope='all mapped convolutions; every fold before IF reset; exact integer shadow verification; original weights and mapping; five 4-bit SCUs and one signed coarse MR per Macro',
        digital=control,shared=result,disagreements=disagreements,layers=layers,
        global_minimum=lo,global_maximum=hi,global_required_signed_bits=signed_bits(lo,hi),
        generic_bounds=[guaranteed_bound(t,f) for t in [16,64] for f in [1,2,4,8]])
    out=args.output
    out.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({name:{k:v[k] for k in ['minimum','maximum','required_signed_bits','folds']} for name,v in layers.items()},indent=2))
    print('Global signed bits:',signed_bits(lo,hi),'; prediction disagreements:',disagreements,flush=True)


if __name__=='__main__': main()
