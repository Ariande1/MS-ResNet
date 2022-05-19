import argparse
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models.MS_ResNet
import torchvision.datasets as datasets

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-weights',
                        type=str,
                        default="resnet34.pth",
                        help='the weights file you want to test')
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-b',
                        type=int,
                        default=100,
                        help='batch size for dataloader')
    args = parser.parse_args()
    if args.net == "resnet34":
        net = models.MS_ResNet.resnet34()
    elif args.net == "resnet104":
        net = models.MS_ResNet.resnet104()
    elif args.net == "resnet18":
        net = models.MS_ResNet.resnet18()

    def get_test_dataloader(batch_size=16, num_workers=4, shuffle=False):
        valdir = "/data1/imagenet/val"
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        ImageNet_test = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        ImageNet_test_loader = DataLoader(ImageNet_test,
                                          shuffle=shuffle,
                                          num_workers=num_workers,
                                          batch_size=batch_size)
        return ImageNet_test_loader

    ImageNet_test_loader = get_test_dataloader(
        num_workers=4,
        batch_size=args.b,
    )

    # net = torch.load(args.weights)
    net.load_state_dict({
        k.replace('module.', ''): v
        for k, v in torch.load(args.weights).items()
    })
    net.cuda()
    net = torch.nn.DataParallel(net)
    net.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    start = time.time()
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(ImageNet_test_loader):
            if (n_iter % 10 == 0):
                print("iteration: {}\ttotal {} iterations".format(
                    n_iter + 1, len(ImageNet_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()

            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            # compute top 5
            correct_5 += correct[:, :5].sum()

            # compute top1
            correct_1 += correct[:, :1].sum()
    finish = time.time()

    print()
    print("Time consumed:", finish - start)
    print("Top 1 acc: ", correct_1.item() / len(ImageNet_test_loader.dataset))
    print("Top 5 acc: ", correct_5.item() / len(ImageNet_test_loader.dataset))
    print("Parameter numbers: {}".format(
        sum(p.numel() for p in net.parameters())))
