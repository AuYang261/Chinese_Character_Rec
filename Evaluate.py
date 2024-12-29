import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader

from Data import MyDataset
from Utils import find_max_log, has_log_file


def evaluate(model, args):
    print("===Evaluate EffNetV2===")
    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.459], std=[0.391]),
        ]
    )

    model.eval()
    if os.path.exists(args.log_root + args.model_name):
        print("Using model: ", args.log_root + args.model_name)
        checkpoint = torch.load(args.log_root + args.model_name)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif has_log_file(args.log_root):
        file = find_max_log(args.log_root)
        print("Using log file: ", file)
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Warning: No log file")

    model.to(torch.device("cuda:0"))
    test_loader = DataLoader(
        MyDataset(
            args.data_root + "test.txt",
            num_class=args.num_classes,
            transforms=transform,
            gray=True
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    total = 0.0
    correct = 0.0
    if os.path.exists("error"):
        os.system("rm -rf error")
    os.makedirs("error")
    print("Evaluating...")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].cuda(), data[1].cuda()
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
            print("\r total: ", total, "/", len(test_loader.dataset), correct / total * 100, "%", end="")
            error_index = (predict != labels).nonzero()
            # save error image
            for index in error_index:
                img = inputs[index].cpu().numpy()
                img = img[0] * 0.391 + 0.459
                img = img * 255
                img = img.astype("uint8")
                img = img.transpose(1, 2, 0)
                img = img.squeeze()
                img = transforms.ToPILImage()(img)
                img.save("error/" + str(labels[index].item()) + "_" + str(predict[index].item()) + ".png")
    acc = correct / total * 100
    print("\nAccuracy" ": ", acc, "%")
