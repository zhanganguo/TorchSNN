import torch


class LeNet(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.conv1 = torch.nn.Conv2d(in_channels=input_shape[-1], out_channels=16, kernel_size=(5, 5), padding=0,
                                     bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5), padding=0, bias=False)
        flatten_features = 64 * ((input_shape[0] - 4) // 2 - 4) // 2 * ((input_shape[1] - 4) // 2 - 4) // 2
        self.linear1 = torch.nn.Linear(in_features=flatten_features, out_features=128, bias=False)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.ReLU()(torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x))
        x = self.conv2(x)
        x = torch.nn.ReLU()(torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))(x))
        x = x.view(-1, self.num_flat_features(x))
        x = torch.nn.ReLU()(self.linear1(x))
        x = self.linear2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class FCN(torch.nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=in_features, out_features=800, bias=False)
        self.linear2 = torch.nn.Linear(in_features=800, out_features=800, bias=False)
        self.linear3 = torch.nn.Linear(in_features=800, out_features=num_classes, bias=False)

        self.in_features = in_features
        self.num_classes = num_classes

    def forward(self, x):
        x = torch.nn.ReLU()(self.linear1(x))
        x = torch.nn.ReLU()(self.linear2(x))
        x = self.linear3(x)
        return x


class Cifar10Net(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.conv1 = torch.nn.Conv2d(in_channels=input_shape[-1], out_channels=32, kernel_size=(3, 3), padding=1,
                                     bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), dilation=1, stride=2, padding=1, bias=False)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), dilation=1, stride=2, padding=1, bias=False)
        flatten_features = 128 * (input_shape[0] // 2) // 2 * (input_shape[1] // 2) // 2
        self.linear1 = torch.nn.Linear(in_features=flatten_features, out_features=256, bias=False)
        self.linear2 = torch.nn.Linear(in_features=256, out_features=num_classes, bias=False)

    def forward(self, x):
        x = torch.nn.ReLU()(self.conv1(x))
        x = torch.nn.ReLU()(self.conv2(x))
        x = torch.nn.ReLU()(self.conv3(x))
        x = torch.nn.ReLU()(self.conv4(x))
        x = x.view(-1, self.num_flat_features(x))
        x = torch.nn.ReLU()(torch.nn.Dropout(p=0.4)(self.linear1(x)))
        x = self.linear2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
