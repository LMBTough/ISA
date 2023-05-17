# from utils import CausalMetric, VisionSensitivityN
from evaluation import CausalMetric
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50,inception_v3,googlenet,vgg16,mobilenet_v2
from saliency.saliency_zoo import fast_ig, guided_ig, big, ma2ba_smooth,ma2ba_sharp, f1, f2,fourier,agi,ig,our
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
                    choices=['fast_ig', 'guided_ig', 'big', 'ma2ba', 'agi', 'our', 'f1', 'f2', 'fourier', 'ig','our2',"saliencymap","our_rev"])

args = parser.parse_args()

model = eval(f"{args.model}(pretrained=True).eval().to(device)")
sm = nn.Softmax(dim=-1)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
norm_layer = transforms.Normalize(mean, std)
sm = nn.Softmax(dim=-1)
model = nn.Sequential(norm_layer, model, sm).to(device)

deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)
insertion = CausalMetric(model, 'ins', 224, substrate_fn=torch.zeros_like)

if __name__ == "__main__":

    if args.attr_method == 'fast_ig':
        attribution = np.load(f"attributions/{args.model}_fast_ig_attributions.npy")
    elif args.attr_method == 'guided_ig':
        attribution = np.load(f"attributions/{args.model}_guided_ig_attributions.npy")
    elif args.attr_method == 'big':
        attribution = np.load(f"attributions/{args.model}_big_attributions.npy")
    elif args.attr_method == 'ma2ba':
        attribution = np.load(f"attributions/{args.model}_ma2ba_attributions.npy")
    elif args.attr_method == 'agi':
        attribution = np.load(f"attributions/{args.model}_agi_attributions.npy")
    elif args.attr_method == 'our':
        attribution = np.load(f"attributions/{args.model}_our_attributions.npy")
    elif args.attr_method == 'our2':
        attribution = np.load(f"attributions/{args.model}_our2_attributions.npy")
    elif args.attr_method == 'f1':
        attribution = np.load(f"attributions/{args.model}_f1_attributions.npy")
    elif args.attr_method == 'f2':
        attribution = np.load(f"attributions/{args.model}_f2_attributions.npy")
    elif args.attr_method == 'fourier':
        attribution = np.load(f"attributions/{args.model}_fourier_attributions.npy")
    elif args.attr_method == 'ig':
        attribution = np.load(f"attributions/{args.model}_ig_attributions.npy")
    elif args.attr_method == 'saliencymap':
        attribution = np.load(f"attributions/{args.model}_saliencymap_attributions.npy")
    elif args.attr_method == 'our_rev':
        attribution = np.load(f"attributions/{args.model}_our_rev_attributions.npy")
    scores = {'del': deletion.evaluate(
        img_batch, attribution, 100), 'ins': insertion.evaluate(img_batch, attribution, 100)}
    # for idx, x_ in tqdm(enumerate(img_batch),total=1000):
    #     # attribution = calculate_attribution(model,x_,label_.unsqueeze(0)).cpu().detach().numpy().squeeze()
    #     h1 = insertion.single_run(x_.unsqueeze(
    #         0).cpu(), attribution[idx:idx+1], verbose=0)
    #     # print("insertion:",h1.mean() / h1[-1])
    #     h2 = deletion.single_run(x_.unsqueeze(
    #         0).cpu(), attribution[idx:idx+1], verbose=0)
    #     # print("deletion:",h2.mean() / h1[-1])
    #     scores['ins'].append(h1)
    #     scores['del'].append(h2)
    scores['ins'] = np.array(scores['ins'])
    scores['del'] = np.array(scores['del'])
    np.savez(f"scores/{args.model}_{args.attr_method}_scores.npz", **scores)
