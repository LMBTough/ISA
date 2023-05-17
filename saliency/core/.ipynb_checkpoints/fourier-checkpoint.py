import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import torch
def dct1(x):
    """
    Discrete Cosine Transform, Type I
    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.fft.fft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1).real.view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I
    Our definition if idct1 is such that idct1(dct1(x)) == x
    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape).real


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)

def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_3d(dct_3d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)




class FGSMGrad:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max

    def __call__(self, model, data, target, num_steps=100, alpha=0.003, early_stop=True, use_sign=False, use_softmax=False):
        dt_fft = data.clone().cpu().detach().numpy()
        dt_fft = torch.from_numpy(dt_fft).to(device)
        dt_fft = dct_2d(dt_fft)
        dt_fft = torch.nn.Parameter(dt_fft)
        dt_fft.requires_grad_(True)
        target_clone = target.clone()
        hats = [[dt_fft[i:i+1].clone()] for i in range(dt_fft.shape[0])]
        hats_ifft = [[data[i:i+1].clone()] for i in range(dt_fft.shape[0])]
        grads = [[] for _ in range(dt_fft.shape[0])]
        grads_ifft = [[] for _ in range(dt_fft.shape[0])]
        leave_index = np.arange(dt_fft.shape[0])
        for _ in range(num_steps):
            dt = idct_2d(dt_fft).float()
            output = model(dt)
            model.zero_grad()
            if use_softmax:
                tgt_out = torch.diag(
                    F.softmax(output, dim=-1)[:, target]).unsqueeze(-1)
            else:
                tgt_out = torch.diag(output[:, target]).unsqueeze(-1)
            tgt_out.sum().backward()
            grad = dt_fft.grad.detach()
            for i, idx in enumerate(leave_index):
                grads[idx].append(grad[i:i+1].clone())
                grads_ifft[idx].append(idct_2d(grad[i:i+1]).clone())
            if use_sign:
                data_grad = dt_fft.grad.detach().sign()
                adv_data = dt_fft - alpha * data_grad
                dt_fft.data = adv_data
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt_fft[i:i+1].data.clone())
                    hats_ifft[idx].append(idct_2d(dt_fft[i:i+1]).clone())
            else:
                data_grad = grad / \
                    grad.reshape(grad.shape[0], -1).norm(dim=1,
                                                      keepdim=True).view(-1, 1, 1, 1)
                adv_data = dt_fft - alpha * data_grad * 100
                dt_fft.data = adv_data
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt_fft[i:i+1].data.clone())
                    hats_ifft[idx].append(idct_2d(dt_fft[i:i+1]).clone())
            if early_stop:
                dt = idct_2d(dt_fft).float()
                adv_pred = model(dt)
                adv_pred_argmax = adv_pred.argmax(-1)
                removed_index = np.where((adv_pred_argmax != target).cpu())[0]
                keep_index = np.where((adv_pred_argmax == target).cpu())[0]
                if len(keep_index) == 0:
                    break
                if len(removed_index) > 0:
                    # dt = dt[keep_index, :].detach().requires_grad_(True)
                    dt_fft = dt_fft[keep_index,
                                    :].detach().requires_grad_(True)
                    data = data[keep_index, :]
                    target = target[keep_index]
                    leave_index = leave_index[keep_index]
        dt_fft = [hat[-1] for hat in hats]
        dt_fft = torch.cat(dt_fft, dim=0).requires_grad_(True)
        dt = idct_2d(dt_fft).float()
        adv_pred = model(dt)
        model.zero_grad()
        if use_softmax:
            tgt_out = torch.diag(F.softmax(adv_pred, dim=-1)
                                 [:, target_clone]).unsqueeze(-1)
        else:
            tgt_out = torch.diag(adv_pred[:, target_clone]).unsqueeze(-1)
        tgt_out.sum().backward()
        grad = dt_fft.grad.detach()
        for i in range(grad.shape[0]):
            grads[i].append(grad[i:i+1].clone())
            grads_ifft[i].append(idct_2d(grad[i:i+1]).clone())
        hats = [torch.cat(hat, dim=0) for hat in hats]
        hats_ifft = [torch.cat(hat, dim=0) for hat in hats_ifft]
        grads = [torch.cat(grad, dim=0) for grad in grads]
        grads_ifft = [torch.cat(grad, dim=0) for grad in grads_ifft]
        success = adv_pred.argmax(-1) != target_clone
        return dt, success, adv_pred, hats, grads, hats_ifft, grads_ifft


class Ma2Ba:
    def __init__(self, model, type="1"):
        self.model = model
        self.type = type

    def __call__(self, hats, grads):
        t_list = hats[1:] - hats[:-1]
        if self.type == "1":
            grads = grads[:-1]
        else:
            grads = (grads[:-1] + grads[1:]) / 2
        total_grads = -torch.sum(t_list * grads, dim=0)
        attribution_map = total_grads.unsqueeze(0)
        return attribution_map.detach().cpu().numpy()