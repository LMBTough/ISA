from saliency.core import FastIG, GuidedIG, pgd_step, BIG, FGSM, Ma2Ba, SaliencyGradient,SmoothGradient,DL,FGSMGrad,IntegratedGradient,dct_2d,idct_2d,DI,gkern,Ma2BaF,FGSMGradF,exp
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
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    method = FastIG(model)
    result = method(data, target).squeeze()
    return np.expand_dims(result, axis=0)


def guided_ig(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
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

    result =  method.GetMask(
        im, call_model_function, call_model_args, x_steps=15, x_baseline=baseline)
    return np.expand_dims(result, axis=0)


def agi(model, data, target, epsilon=0.05, max_iter=20, topk=20):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    random.seed(3407)
    selected_ids = random.sample(list(range(0, 999)), topk)
    output = model(data)
    # get the index of the max log-probability
    # init_pred = output.max(1, keepdim=True)[1]
    init_pred = output.argmax(-1)

    top_ids = selected_ids  # only for predefined ids
    # initialize the step_grad towards all target false classes
    step_grad = 0
    # num_class = 1000 # number of total classes
    for l in top_ids:
        # targeted = torch.tensor([l]).to(device)
        targeted = torch.tensor([l] * data.shape[0]).to(device)
        # if targeted.item() == init_pred.item():
        #     if l < 999:
        #         # replace it with l + 1
        #         targeted = torch.tensor([l+1]).to(device)
        #     else:
        #         # replace it with l + 1
        #         targeted = torch.tensor([l-1]).to(device)
        #     # continue # we don't want to attack to the predicted class.
        if l < 999:
            targeted[targeted == init_pred] = l + 1
        else: 
            targeted[targeted == init_pred] = l - 1

        delta, perturbed_image = pgd_step(
            data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    adv_ex = step_grad.squeeze().detach().cpu().numpy()  # / topk
    return adv_ex

def big(model, data, target, data_min=0, data_max=1, epsilons=[36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255], class_num=1000, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    attacks = [FGSM(eps, data_min, data_max) for eps in epsilons]
    big = BIG(model, attacks, class_num)
    attribution_map, success = big(model, data, target, gradient_steps)
    return attribution_map


def ma2ba_smooth(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255,use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    ma2ba = Ma2Ba(model)
    attack = FGSMGrad(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    input_baseline, success, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(ma2ba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map

def ma2ba_sharp(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255,use_sign=False, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    ma2ba = Ma2Ba(model)
    attack = FGSMGrad(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    input_baseline, success, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(ma2ba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map

def ig(model, data, target, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    ig = IntegratedGradient(model)
    return ig(data, target, gradient_steps=gradient_steps)

def sm(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sm = SaliencyGradient(model)
    return sm(data, target)

def sg(model, data, target,stdevs=0.15, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sg = SmoothGradient(model,stdevs=stdevs)
    return sg(data, target, gradient_steps=gradient_steps)

def deeplift(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    dl = DL(model)
    return dl(data, target)

def saliencymap(model,data,target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    saliencymap = Saliency(model)
    return saliencymap.attribute(data, target).cpu().detach().numpy()

def f1(model,data,target, min=0, max=1):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    
    def clip_by_tensor(t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    T_kernel = gkern(7, 3)
    image_width = 224
    momentum = 1.0
    num_iter = 10
    eps = 16.0 / 255.0
    alpha = eps / num_iter
    x = data.clone()
    grad = 0
    
    rho = 0.5
    N = 20
    sigma = 16.0
    
    total = torch.zeros_like(data)
    for i in range(num_iter):
        noise = 0
        # 加一条搜索路径
        # inner_x = x.clone().detach().requires_grad_(True)
        for k in range(4):
            inner_grad = 0
            inner_x = x.clone()
            inner_x_dct = dct_2d(inner_x)
            for n in range(5):
                # print('-------------', torch.from_numpy(caluculate_c()).permute(0, 3, 1, 2).shape)
                # gauss = torch.randn(x.size()[0], 3, image_width, image_width) * (sigma / 255)
                gauss = torch.randn(x.size()[0], 3, image_width, image_width)
                gauss = torch.clamp(gauss, -1, 1) * (sigma / 255)
                gauss = gauss.cuda()
                ## 新加代码，等会尝试对频率进攻
                gauss_dct = dct_2d(gauss)
                mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
                inner_x_dct = V(inner_x_dct, requires_grad = True)
                inner_x_idct = idct_2d((inner_x_dct * mask +gauss_dct) )
                # inner_x_idct = V(inner_x_idct, requires_grad = True)

    #             if 'DI' in opt.method:
                di = DI(inner_x_idct)
                output_v3 = model(di)

    #             else:
                # output_v3 = model(inner_x_idct)

                loss = F.cross_entropy(output_v3, target)
                loss.backward(retain_graph=True )
                inner_noise = inner_x_dct.grad.data
    #             if 'TI' in opt.method:
    #                 inner_noise = F.conv2d(inner_noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

    #             # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
    #             if 'MI' in opt.method:
                # inner_noise = inner_noise / torch.abs(inner_noise).mean([1, 2, 3], keepdim=True)
                inner_noise = 0.5 * inner_grad + 0.5 * inner_noise
                inner_grad = inner_noise
    #             else:
                # inner_noise = inner_noise 
                inner_x_dct = inner_x_dct + 0.005 * torch.sign(inner_noise)
                # inner_x = clip_by_tensor(inner_x, min, max)
                # print(torch.autograd.grad(loss, inner_x_idct)[0])
                noise = noise +  torch.autograd.grad(loss, inner_x_idct)[0]
                


        # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        # if 'TI' in opt.method:
        noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        # if 'MI' in opt.method:
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        # else:
        #     noise = noise / N

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
        total += alpha * torch.sign(noise) * noise
    return total.cpu().detach().numpy()

def f2(model,data,target, min=0, max=1):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    """
    The attack algorithm of our proposed Spectrum Simulate Attack
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    T_kernel = gkern(7, 3)
    
    def clip_by_tensor(t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result
    
    image_width = 224
    momentum = 1.0
    num_iter = 10
    eps = 16.0 / 255.0
    alpha = eps / num_iter
    x = data.clone()
    grad = 0
    rho = 0.5
    N = 20
    sigma = 16.0

    total = torch.zeros_like(data)
    for i in range(num_iter):
        noise = 0
        for n in range(N):
            gauss = torch.randn(x.size()[0], 3, image_width, image_width) * (sigma / 255)
            gauss = gauss.cuda()
            x_dct = dct_2d(x + gauss).cuda()
            mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
            x_idct = idct_2d(x_dct * mask)
            x_idct = V(x_idct, requires_grad = True)

            # DI-FGSM https://arxiv.org/abs/1803.06978
            output_v3 = model(DI(x_idct))

            output_v3 = model(x_idct)
            loss = F.cross_entropy(output_v3, target)
            loss.backward()
            noise += x_idct.grad.data
        noise = noise / N

        # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
        total += alpha * torch.sign(noise) * noise
    return total.cpu().detach().numpy()

def fourier(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255,return_both=False):
    ma2ba = Ma2BaF(model)
    attack = FGSMGradF(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, success, _, hats, grads, hats_ifft, grads_ifft = attack(
        model, data, target, use_sign=False, use_softmax=False)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(ma2ba(hats_ifft[i], grads_ifft[i]))
    if return_both:
        attribution_map_freq = list()
        for i in range(len(hats)):
            attribution_map_freq.append(ma2ba(hats[i], grads[i]))
        attribution_map_freq = np.concatenate(attribution_map_freq, axis=0)
    attribution_map = np.concatenate(attribution_map, axis=0)
    if return_both:
        return attribution_map,attribution_map_freq
    return attribution_map


def our(model,x,label,step_size=5000,add_steps=8,minus_steps=8,alpha=0.004,factor=1.3):
    mask = torch.ones_like(x,dtype=torch.long) # 初始化mask保留所有像素
    importance = torch.zeros_like(x.unsqueeze(0)) # 初始化importance
    n_steps = np.array(x.size()[1:]).prod() // step_size + 1
    removed_count = 0
    for i in tqdm(range(n_steps)):
        combine, combine_flatten = exp(model, x, label,mask, add_steps=add_steps, minus_steps=minus_steps, alpha=alpha, lambda_r=0.01, method="total*delta")
        combine[mask.float().cpu() == 0] = -np.inf # 将已经去掉的设置成无限小
        combine_flatten = np.concatenate([c.flatten()[np.newaxis,:] for c in combine])
        # combine_flatten = combine.flatten()
        
        if removed_count + step_size > combine_flatten.shape[-1]:
            step_size = len(combine_flatten) - removed_count
        m = np.zeros_like(combine_flatten)
        temp = np.argsort(combine_flatten)[:,removed_count:removed_count+step_size]
        for t in range(len(temp)):
            m[t,temp[t]] = 1 
        m = m.reshape(combine.shape).astype(bool)
        a = combine[m]
        # raise NotImplementedError
        if len(a) == 0:
            break
        a = a - a.min(axis=0)
        a = a / (a.max(axis=0)+1e-6) * factor
        importance[:,m.squeeze()] = i + torch.from_numpy(a).cuda() # 设置重要度，从1开始
        m = ~m # 由于m中True是去掉的所以得取反
        m = m.astype(int)

        mask = mask * torch.from_numpy(m).long().to(device) # 把去掉的设置为0
        removed_count += step_size
    # importance[importance == 0] = importance.max() + 1 # 最后剩余的设置为最大的
    # print(torch.sum(importance[importance == 0]))
    # print(torch.sum(importance[importance != 0]))
    return importance.cpu().detach().numpy().squeeze()

def our_rev(model,x,label,step_size=5000,add_steps=8,minus_steps=8,alpha=0.004,factor=1.3):
    mask = torch.ones_like(x,dtype=torch.long) # 初始化mask保留所有像素
    importance = torch.zeros_like(x.unsqueeze(0)) # 初始化importance
    n_steps = np.array(x.size()[1:]).prod() // step_size + 1
    removed_count = 0
    for i in tqdm(range(n_steps)):
        combine, combine_flatten = exp(model, x, label,mask, add_steps=add_steps, minus_steps=minus_steps, alpha=alpha, lambda_r=0.01, method="total*delta")
        combine[mask.float().cpu() == 0] = np.inf # 将已经去掉的设置成无限大
        combine_flatten = np.concatenate([c.flatten()[np.newaxis,:] for c in combine])
        # combine_flatten = combine.flatten()
        
        if removed_count + step_size > combine_flatten.shape[-1]:
            step_size = len(combine_flatten) - removed_count
        m = np.zeros_like(combine_flatten)
        temp = np.argsort(combine_flatten)[::-1][:,combine_flatten.shape[1]-removed_count-step_size:combine_flatten.shape[1]-removed_count]
        for t in range(len(temp)):
            m[t,temp[t]] = 1 
        m = m.reshape(combine.shape).astype(bool)
        a = combine[m]
        # raise NotImplementedError
        if len(a) == 0:
            break
        a = a - a.min(axis=0)
        if i != n_steps-1:
            a = a / (a.max(axis=0)+1e-6) * factor - 0.3
        else:
            a = a / (a.max(axis=0)+1e-6) 
        importance[:,m.squeeze()] = n_steps -i - 1 + torch.from_numpy(a).cuda() # 设置重要度，从1开始
        m = ~m # 由于m中True是去掉的所以得取反
        m = m.astype(int)

        mask = mask * torch.from_numpy(m).long().to(device) # 把去掉的设置为0
        removed_count += step_size
    # importance[importance == 0] = importance.max() + 1 # 最后剩余的设置为最大的
    # print(torch.sum(importance[importance == 0]))
    # print(torch.sum(importance[importance != 0]))
    return importance.cpu().detach().numpy().squeeze()


def our2(model,x,label,step_size=5000):
    mask = torch.ones_like(x,dtype=torch.long) # 初始化mask保留所有像素
    importance = torch.zeros_like(x.unsqueeze(0)) # 初始化importance
    n_steps = np.array(x.size()[1:]).prod() // step_size + 1
    removed_count = 0
    for i in tqdm(range(n_steps)):
        combine, combine_flatten = exp(model, x, label,mask, add_steps=8, minus_steps=8, alpha=0.004, lambda_r=0.01, method="total*delta")
        combine[mask.float().cpu() == 0] = -np.inf # 将已经去掉的设置成无限小
        combine_flatten = np.concatenate([c.flatten()[np.newaxis,:] for c in combine])
        # combine_flatten = combine.flatten()
        
        if removed_count + step_size > combine_flatten.shape[-1]:
            step_size = len(combine_flatten) - removed_count
        m = np.zeros_like(combine_flatten)
        temp = np.argsort(combine_flatten)[:,removed_count:removed_count+step_size]
        for t in range(len(temp)):
            m[t,temp[t]] = 1 
        m = m.reshape(combine.shape).astype(bool)
        combine, combine_flatten = exp(model, x, label,m, add_steps=8, minus_steps=8, alpha=0.004, lambda_r=0.01, method="total*delta")
        a = combine[m]
        # raise NotImplementedError
        if len(a) == 0:
            break
        a = a - a.min(axis=0)
        a = a / (a.max(axis=0)+1e-6) * 1.3
        importance[:,m.squeeze()] = i + torch.from_numpy(a).cuda() # 设置重要度，从1开始
        m = ~m # 由于m中True是去掉的所以得取反
        m = m.astype(int)

        mask = mask * torch.from_numpy(m).long().to(device) # 把去掉的设置为0
        removed_count += step_size
    # importance[importance == 0] = importance.max() + 1 # 最后剩余的设置为最大的
    # print(torch.sum(importance[importance == 0]))
    # print(torch.sum(importance[importance != 0]))
    return importance.cpu().detach().numpy().squeeze()
