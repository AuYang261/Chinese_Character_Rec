import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from EfficientNetV2_gray.Evaluate import evaluate
from EfficientNetV2_gray.Train import train
from EfficientNetV2_gray.demo import demo
from Utils import classes_txt
from Data import to_gray

parser = argparse.ArgumentParser(description="EfficientNetV2 arguments")
parser.add_argument(
    "--mode", dest="mode", type=str, default="train", help="Mode of net"
)
parser.add_argument(
    "--epoch", dest="epoch", type=int, default=50, help="Epoch number of training"
)
parser.add_argument(
    "--batch_size", dest="batch_size", type=int, default=512, help="Value of batch size"
)
parser.add_argument("--lr", dest="lr", type=float, default=0.0001, help="Value of lr")
parser.add_argument(
    "--img_size", dest="img_size", type=int, default=32, help="reSize of input image"
)
parser.add_argument(
    "--data_root", dest="data_root", type=str, default="../data/", help="Path to data"
)
parser.add_argument(
    "--log_root", dest="log_root", type=str, default="../log_gray/", help="Path to log0.pth"
)
parser.add_argument(
    "--model",
    dest="model_name",
    type=str,
    default=".pth",
    help="Path to model.pth",
)
parser.add_argument(
    "--num_classes",
    dest="num_classes",
    type=int,
    default=3755,
    help="Classes of character",
)
parser.add_argument(
    "--demo_img",
    dest="demo_img",
    type=str,
    default="../asserts/pei.png",
    help="Path to demo image",
)
parser.add_argument("--scale", dest="scale", type=str, default="l", help="s/m/l")
args = parser.parse_args()


if __name__ == "__main__":
    args.log_root = args.log_root.replace("log", "log_" + args.scale)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    gray_data_root = os.path.join(args.data_root, "gray/")
    if not os.path.exists(gray_data_root):
        os.makedirs(gray_data_root)
        for t in ["train/", "test/"]:
            for d in os.listdir(args.data_root + t):
                if os.path.isdir(args.data_root + t + d):
                    for f in os.listdir(args.data_root + t + d):
                        to_gray(args.data_root + t + d + "/" + f, gray_data_root + t + d + "/" + f)
    args.data_root = gray_data_root
        
    if not os.path.exists(args.data_root + "train.txt"):
        classes_txt(
            args.data_root + "train", args.data_root + "train.txt", args.num_classes
        )
    if not os.path.exists(args.data_root + "test.txt"):
        classes_txt(
            args.data_root + "test", args.data_root + "test.txt", args.num_classes
        )

    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "demo":
        demo(args)
    else:
        print("Unknown mode")
