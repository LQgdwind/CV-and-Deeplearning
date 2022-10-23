import torchvision

# 调用torchvision自带的resnet模型用作对比
def resnet_18():
    return torchvision.models.resnet18(pretrained=False)


def resnet_50():
    return torchvision.models.resnet50(pretrained=False)
