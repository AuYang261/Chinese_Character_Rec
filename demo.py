import pickle

import os

import torch
import torchvision.transforms as transforms
from PIL import Image


from Utils import find_max_log, has_log_file


def demo(model, args):
    print("==Demo EfficientNetV2===")
    print("Input Image: ", args.demo_img)
    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.459], std=[0.391]),
        ]
    )
    img = Image.open(args.demo_img).convert("L")
    img = transform(img)
    img = img.unsqueeze(0)
    model.eval()
    if os.path.exists(args.log_root + args.model_name):
        checkpoint = torch.load(args.log_root + args.model_name)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif has_log_file(args.log_root):
        file = find_max_log(args.log_root)
        print("Using log file: ", file)
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Warning: No model/log file")

    with torch.no_grad():
        output = model(img)
    _, pred = torch.max(output.data, 1)
    f = open("../char_dict", "rb")
    dic = pickle.load(f)
    for cha in dic:
        if dic[cha] == int(pred):
            print("predict: ", cha)
    f.close()
