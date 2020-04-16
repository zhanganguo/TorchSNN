import numpy as np
import torch
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset

from ann.model import LeNet
from ann.training_and_test import train_ann, test_ann
from snn.spiking_neuron import IFParameters
from snn.model import SpikingLeNet
from snn.training_and_test import test_snn


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    simulation_config = {
        'use_gpu': torch.cuda.is_available(),
        'seq_length': 200,
        'max_firing_rate': 500,
        'dt': 0.001
    }
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    #                    torchvision.transforms.RandomCrop(size=[28,28], padding=4),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0], [1]),
                ]
            ),
        ),
        batch_size=32,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(
            root="data",
            train=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0], [1]),
                ]
            ),
        ),
        batch_size=32
    )
    cnn = LeNet(input_shape=(28, 28, 1), num_classes=10)
    cnn = cnn.cuda() if simulation_config['use_gpu'] else cnn

    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.005)
    lr_sch = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
    criterion = torch.nn.CrossEntropyLoss()

    # for i in range(20):
    #     train_ann(model=cnn, train_loader=train_loader, optimizer=optimizer, criterion=criterion, epoch=i+1, simulation_config=simulation_config)
    #
    #     test_ann(model=cnn, test_loader=test_loader, criterion=criterion, simulation_config=simulation_config)
    #     lr_sch.step()
    # torch.save(cnn.state_dict(), 'model_output/LeNet_fashionmnist_params.pkl')

    device = torch.device('cuda')
    scnn = SpikingLeNet(input_shape=(28, 28, 1), num_classes=10, if_param=IFParameters(), device=device,
                        seq_length=simulation_config['seq_length'],
                        max_firing_rate=simulation_config['max_firing_rate'],
                        dt=simulation_config['dt']).to(device)
    scnn.load_state_dict(torch.load('model_output/LeNet_fashionmnist_params.pkl'))
    test_snn(model=scnn, test_loader=test_loader, simulation_config=simulation_config)
