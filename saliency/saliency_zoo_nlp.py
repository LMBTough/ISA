from saliency.core import FastIG, GuidedIG, pgd_step, BIG, FGSM, SaliencyGradient, SmoothGradient, DL, IntegratedGradient, DI, gkern, exp
from captum.attr import Saliency
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
import random
from torch.autograd import Variable as V
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fast_ig(model, data, target):
    
    method = FastIG(model)
    result = method(data, target).squeeze()
    return np.expand_dims(result, axis=0)


def guided_ig(model, data, target):
    model = model[:2]
    
    class_idx_str = 'class_idx_str'

    def call_model_function(images, call_model_args=None, expected_keys=None):
        target_class_idx = call_model_args[class_idx_str]
        images = torch.from_numpy(images).float().to(device)
        images = images.requires_grad_(True)
        output = model(images)
        m = torch.nn.Softmax(dim=1)
        output = m(output)
        outputs = output[:, target_class_idx]
        grads = torch.autograd.grad(
            outputs, images, grad_outputs=torch.ones_like(outputs))[0]
        gradients = grads.cpu().detach().numpy()
        return {'INPUT_OUTPUT_GRADIENTS': gradients}

    im = data.squeeze().cpu().detach().numpy()
    call_model_args = {class_idx_str: target}
    baseline = np.zeros(im.shape)
    method = GuidedIG()

    result = method.GetMask(
        im, call_model_function, call_model_args, x_steps=15, x_baseline=baseline)
    return np.expand_dims(result, axis=0)


def agi(model, data, target, epsilon=0.05, max_iter=20, topk=20):
    model = model[:2]
    
    random.seed(3407)
    selected_ids = random.sample(list(range(0, 999)), topk)
    output = model(data)

    init_pred = output.argmax(-1)

    top_ids = selected_ids

    step_grad = 0

    for l in top_ids:

        targeted = torch.tensor([l] * data.shape[0]).to(device)

        if l < 999:
            targeted[targeted == init_pred] = l + 1
        else:
            targeted[targeted == init_pred] = l - 1

        delta, perturbed_image = pgd_step(
            data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    adv_ex = step_grad.squeeze().detach().cpu().numpy()
    return adv_ex


def big(model, data, target, data_min=0, data_max=1, epsilons=[36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255], class_num=1000, gradient_steps=50):
    
    attacks = [FGSM(eps, data_min, data_max) for eps in epsilons]
    big = BIG(model, attacks, class_num)
    attribution_map, success = big(model, data, target, gradient_steps)
    return attribution_map


def ig(model, data, target, gradient_steps=50):
    
    ig = IntegratedGradient(model)
    return ig(data, target, gradient_steps=gradient_steps)


def sm(model, data, target):
    
    sm = SaliencyGradient(model)
    return sm(data, target)


def sg(model, data, target, stdevs=0.15, gradient_steps=50):
    
    sg = SmoothGradient(model, stdevs=stdevs)
    return sg(data, target, gradient_steps=gradient_steps)


def deeplift(model, data, target):
    
    dl = DL(model)
    return dl(data, target)


def saliencymap(model, data, target):
    
    saliencymap = Saliency(model)
    return saliencymap.attribute(data, target).cpu().detach().numpy()


def ISA(model, x, label, step_size=5000, add_steps=8, minus_steps=8, alpha=0.004, factor=1.3):
    mask = torch.ones_like(x, dtype=torch.long)
    importance = torch.zeros_like(x.unsqueeze(0))
    n_steps = np.array(x.size()[1:]).prod() // step_size + 1
    removed_count = 0
    for i in tqdm(range(n_steps)):
        combine, combine_flatten = exp(model, x, label, mask, add_steps=add_steps,
                                       minus_steps=minus_steps, alpha=alpha, lambda_r=0.01, method="total*delta")
        combine[mask.float().cpu() == 0] = -np.inf
        combine_flatten = np.concatenate(
            [c.flatten()[np.newaxis, :] for c in combine])

        if removed_count + step_size > combine_flatten.shape[-1]:
            step_size = len(combine_flatten) - removed_count
        m = np.zeros_like(combine_flatten)
        temp = np.argsort(combine_flatten)[
            :, removed_count:removed_count+step_size]
        for t in range(len(temp)):
            m[t, temp[t]] = 1
        m = m.reshape(combine.shape).astype(bool)
        a = combine[m]

        if len(a) == 0:
            break
        a = a - a.min(axis=0)
        a = a / (a.max(axis=0)+1e-6) * factor
        importance[:, m.squeeze()] = i + torch.from_numpy(a).cuda()
        m = ~m
        m = m.astype(int)

        mask = mask * torch.from_numpy(m).long().to(device)
        removed_count += step_size

    return importance.cpu().detach().numpy().squeeze()
