import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from SNN_ResNet10.models.mr_qat import MRProxy, QATConv2d, bit_planes, LeakageIFNode
from SNN_ResNet10.models.snn_resnet10 import SNNResNet10
from IMC_ResNet.models.cluster_backend import ClusterConv2d, quantize_int5
from IMC_ResNet.models.finetuned_snn import load_finetuned, fuse_explicit_hidden

torch.set_num_threads(2)


def test_shared_proxy_targets_signed_tail_with_exact_forward():
    conv=QATConv2d(36,2,1,bias=False)
    with torch.no_grad(): conv.weight.fill_(-1)
    proxy=MRProxy(conv,target=24,positions=0,shared_bits=6)
    actual=ClusterConv2d(conv,1.)
    x=torch.ones(1,36,1,1)
    for _ in range(8):
        proxy.observe(conv,(x,),conv(x))
        actual(x)
        beta=proxy.mr.new_tensor([1,2,4,8,-16])
        torch.testing.assert_close((proxy.mr.detach()*beta).sum(-1),
                                   (actual.bank.mr.float()*beta).sum(-1),rtol=0,atol=0)
        proxy.fire(None,None,torch.zeros(1,2,1,1))
    assert proxy.shared_min < -32 and proxy.shared_outside > 0
    sum(proxy.losses().values()).backward()
    assert torch.isfinite(conv.weight.grad).all() and conv.weight.grad.abs().sum()>0


@pytest.mark.parametrize('channels', [36, 600])
def test_proxy_matches_actual_bank_across_folds_time_and_resets(channels):
    torch.manual_seed(22)
    conv = QATConv2d(channels, 2, 1, bias=False)
    actual = ClusterConv2d(conv, 1.)
    proxy = MRProxy(conv, target=3, positions=0)
    for t in range(6):
        x = torch.randint(0, 2, (2, channels, 2, 2)).float()
        proxy.observe(conv, (x,), conv(x))
        actual(x)
        torch.testing.assert_close(proxy.mr.detach(), actual.bank.mr.float(), rtol=0, atol=0)
        torch.testing.assert_close(proxy.total.detach().remainder(16), actual.bank.scu.float(), rtol=0, atol=0)
        spike = torch.zeros(2, 2, 2, 2)
        if t % 2:
            spike[0, 0, 0, 0] = 1
        proxy.fire(None, None, spike)
        actual.fire(spike)
        torch.testing.assert_close(proxy.total.detach(), (actual.bank.scu + 16 * actual.bank.mr).float(), rtol=0, atol=0)
    terms = proxy.losses()
    sum(terms.values()).backward()
    assert torch.isfinite(conv.weight.grad).all()
    assert conv.weight.grad.abs().sum() > 0
    proxy.reset()
    assert proxy.total is None and not proxy.overflow_losses


def test_bits_exact_and_gradient_nonzero():
    weight = torch.linspace(-1, 1, 32).reshape(1, 32, 1, 1).requires_grad_()
    planes = bit_planes(weight)
    q, _ = quantize_int5(weight)
    expected = (((q & 31)[..., None] >> torch.arange(5)) & 1).float()
    torch.testing.assert_close(planes, expected, rtol=0, atol=0)
    (planes * torch.arange(1., 6.)).sum().backward()
    assert weight.grad.abs().sum() > 0 and torch.isfinite(weight.grad).all()


def test_observer_does_not_change_output_and_clears_only_firing_lanes():
    conv = QATConv2d(36, 2, 1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(-1)
    proxy = MRProxy(conv, target=2, positions=0)
    x = torch.ones(1, 36, 1, 1)
    before = conv(x)
    hook = conv.register_forward_hook(proxy.observe)
    for _ in range(32):
        torch.testing.assert_close(conv(x), before, rtol=0, atol=0)
        proxy.fire(None, None, torch.zeros(1, 2, 1, 1))
    assert proxy.max_mr == 72 and proxy.ge64 > 0
    proxy.fire(None, None, torch.tensor([[[[1.]], [[0.]]]]))
    assert proxy.total[:, 0].count_nonzero() == 0
    assert proxy.total[:, 1].max() == 1152
    hook.remove()


def test_leakage_loader_and_fused_checkpoint_without_threshold_file(tmp_path):
    from spikingjelly.activation_based import neuron
    source = SNNResNet10().eval()
    path = tmp_path/'model.pt'
    fused = fuse_explicit_hidden(source)
    torch.save({'model': fused.state_dict(), 'hard_reset': True, 'hidden_bn_fused': True,
                'leakage': .99, 'if_thresholds': [2.] * 9}, path)
    restored, metadata = load_finetuned(path, tmp_path/'missing.json')
    nodes = [m for m in restored.modules() if isinstance(m, neuron.IFNode)]
    assert all(isinstance(m, LeakageIFNode) and m.leakage == .99 and m.v_threshold == 2 for m in nodes)
    assert metadata['threshold_source'] == 'checkpoint'
    assert isinstance(fuse_explicit_hidden(restored).layer1.bn1, nn.Identity)
    node = nodes[0]
    node(torch.tensor([1.]))
    node(torch.tensor([0.]))
    torch.testing.assert_close(node.v, torch.tensor([.99]))


def test_sampled_positions_match_actual_subset():
    torch.manual_seed(33)
    conv = QATConv2d(36, 2, 1, bias=False)
    actual = ClusterConv2d(conv, 1.)
    proxy = MRProxy(conv, positions=3)
    x = torch.randint(0, 2, (1, 36, 3, 3)).float()
    for _ in range(4):
        proxy.observe(conv, (x,), conv(x))
        actual(x)
        torch.testing.assert_close(proxy.mr.detach(), actual.bank.mr.index_select(2, proxy.indices).float(), rtol=0, atol=0)
        proxy.fire(None, None, torch.zeros(1, 2, 3, 3))


def test_leakage_train_eval_sequence_equivalence():
    a = LeakageIFNode(leakage=.9, v_threshold=2., v_reset=0.)
    b = copy.deepcopy(a).eval()
    for x in [1., 0., 1.2, 0., -.3, 1.]:
        torch.testing.assert_close(a(torch.tensor([x])), b(torch.tensor([x])), rtol=0, atol=0)
        torch.testing.assert_close(a.v, b.v, rtol=0, atol=0)


def test_leaky_cluster_matches_digital_full_small_network():
    from IMC_ResNet.models.finetuned_snn import make_explicit_cluster, make_explicit_quantized_reference
    from spikingjelly.activation_based import neuron
    source = SNNResNet10(base_channels=2, v_threshold=.7).eval()
    for name, node in list(source.named_modules()):
        if isinstance(node, neuron.IFNode):
            source.set_submodule(name, LeakageIFNode(leakage=.99, v_threshold=.7, v_reset=0.))
    fused = fuse_explicit_hidden(source)
    digital = make_explicit_quantized_reference(fused)
    actual, _ = make_explicit_cluster(fused)
    images = torch.randn(16, 1, 1, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(actual(images), digital(images), atol=1e-5, rtol=1e-5)


def test_materialized_export_reloads_same_forward(tmp_path):
    from types import SimpleNamespace
    from SNN_ResNet10.scripts.finetune_mr_qat import export
    from IMC_ResNet.models.finetuned_snn import explicit_targets
    model = fuse_explicit_hidden(SNNResNet10().eval())
    for name, _, _ in explicit_targets(model):
        conv = model.get_submodule(name)
        qat = QATConv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                        conv.stride, conv.padding, bias=True)
        qat.load_state_dict(conv.state_dict())
        model.set_submodule(name, qat)
    args = SimpleNamespace(leakage=1., steps=64, target=56, positions=32, beta=.05, gamma=.01)
    path = tmp_path/'export.pt'
    export(model, args, .1, [], path)
    loaded, _ = load_finetuned(path, tmp_path/'unused.json')
    images = torch.randn(3, 1, 1, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(model(images), loaded(images), rtol=0, atol=0)


def test_detached_reset_changes_backward_only():
    a = LeakageIFNode(leakage=.99, v_threshold=.7, v_reset=0., detach_reset=False)
    b = LeakageIFNode(leakage=.99, v_threshold=.7, v_reset=0., detach_reset=True)
    x = torch.tensor([.4], requires_grad=True)
    for _ in range(64):
        torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
        torch.testing.assert_close(a.v, b.v, rtol=0, atol=0)
