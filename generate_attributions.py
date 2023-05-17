import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, inception_v3, googlenet, vgg16, mobilenet_v2
from saliency.saliency_zoo import fast_ig, guided_ig, big, ma2ba_smooth,ma2ba_sharp, f1, f2, fourier, agi, ig,our,our2,saliencymap,our_rev
from tqdm import tqdm
import torch
import numpy as np
import argparse
import torch
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(3407)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_batch = torch.load("img_batch.pt")
target_batch = torch.load("label_batch.pt")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='inception_v3',
                    choices=["inception_v3", "resnet50", "googlenet", "vgg16", "mobilenet_v2"])
parser.add_argument('--attr_method', type=str, default='fast_ig',
                    choices=['fast_ig', 'guided_ig', 'big', 'ma2ba', 'f1', 'f2', 'fourier', 'agi', 'ig', 'our','our2',"saliencymap","our_rev"])

args = parser.parse_args()

if args.attr_method == 'fast_ig':
    attr_method = fast_ig
elif args.attr_method == 'guided_ig':
    attr_method = guided_ig
elif args.attr_method == 'big':
    attr_method = big
elif args.attr_method == 'ma2ba':
    attr_method = ma2ba
elif args.attr_method == 'f1':
    attr_method = f1
elif args.attr_method == 'f2':
    attr_method = f2
elif args.attr_method == "fourier":
    attr_method = fourier
elif args.attr_method == "agi":
    attr_method = agi
elif args.attr_method == "ig":
    attr_method = ig
elif args.attr_method == "our":
    attr_method = our
elif args.attr_method == "our2":
    attr_method = our2
elif args.attr_method == "saliencymap":
    attr_method = saliencymap
elif args.attr_method == "our_rev":
    attr_method = our_rev

# model = inception_v3(pretrained=True).eval().to(device)
model = eval(f"{args.model}(pretrained=True).eval().to(device)")
sm = nn.Softmax(dim=-1)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
norm_layer = transforms.Normalize(mean, std)
sm = nn.Softmax(dim=-1)
model = nn.Sequential(norm_layer, model, sm).to(device)

if __name__ == "__main__":
    attributions = []
    if args.attr_method == 'fast_ig' or args.attr_method == 'guided_ig':
        batch_size = 1
    elif args.attr_method == 'big':
        batch_size = 4
    elif args.attr_method == 'agi':
        batch_size = 64
    elif args.attr_method == 'ig':
        batch_size = 4
    elif args.attr_method == 'ma2ba':
        batch_size = 128
    elif args.attr_method == 'f1':
        batch_size = 64
    elif args.attr_method == 'f2':
        batch_size = 64
    elif args.attr_method == 'fourier':
        batch_size = 16
    elif args.attr_method == 'our':
        batch_size = 64
    elif args.attr_method == 'our2':
        batch_size = 64
    elif args.attr_method == 'saliencymap':
        batch_size = 128
    elif args.attr_method == 'our_rev':
        batch_size = 64
    for i in tqdm(range(0, len(img_batch), batch_size)):
        img = img_batch[i:i+batch_size].to(device)
        target = target_batch[i:i+batch_size].to(device)
        attributions.append(attr_method(model, img, target))
    attributions = np.concatenate(attributions, axis=0)
    np.save("attributions/" + args.model+"_" +
            args.attr_method+"_attributions.npy", attributions)
