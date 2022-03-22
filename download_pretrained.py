import torch

if __name__ == '__main__':
    # can also be: "deeplabv3_resnet101", "deeplabv3_mobilenet_v3_large"
    model = torch.hub.load('pytorch/vision:v0.7.0', "deeplabv3_resnet50",
                           pretrained=True)
    model.eval()
