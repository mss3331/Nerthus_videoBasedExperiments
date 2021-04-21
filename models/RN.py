import torchvision.models as models
import torch.nn as nn

class ResNet50_FE(nn.Module):
            def __init__(self,pretrained=True):
                super(ResNet50_FE, self).__init__()
                original_ResNet50 = models.resnet50(pretrained=pretrained)
                # self.activation_36 = nn.Sequential(*list(original_ResNet50.children())[:-1])
                self.activation_0 = nn.Sequential(*list(original_ResNet50.children())[:6])
                print(self.activation_0)
                # self.activation_1 = nn.Sequential(*list(original_ResNet50.children())[6])
                self.activation_1 = nn.Sequential(*list(original_ResNet50.children())[6][:2])
                print(self.activation_1)
                self.activation_2 = nn.Sequential(*list(original_ResNet50.children())[7])
                print(self.activation_2)

                # print(len(list(original_ResNet50.children())))
                # print(list(original_ResNet50.children()))
                # exit(0)
                # self.features = nn.Sequential(
                #     # stop at conv4
                #     *list(original_ResNet50.features.children())[:-3]
                # )
            def forward(self, x):
                # x=self.activation_36(x)
                # print("activation_36",x.shape)
                x= self.activation_0(x)
                print("activation_0",x.shape)
                x = self.activation_1(x)
                print("activation_1", x.shape)
                x = self.activation_2(x)
                print("activation_2", x.shape)

                return x