import numpy as np
import torch
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset

from ann.model import FCN
from ann.training_and_test import train_ann, test_ann
from snn.spiking_neuron import IFParameters
from snn.model import SpikingFCN
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
    fcn = FCN(in_features=28 * 28 * 1, num_classes=10)
    fcn = fcn.cuda() if simulation_config['use_gpu'] else fcn

    optimizer = torch.optim.Adam(fcn.parameters(), lr=0.005)
    lr_sch = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
    criterion = torch.nn.CrossEntropyLoss()

    # for i in range(10):
    #     train_ann(model=fcn, train_loader=train_loader, optimizer=optimizer, criterion=criterion, epoch=i + 1,
    #           config=config)
    #
    #     test_ann(model=fcn, test_loader=test_loader, criterion=criterion, config=config)
    #     lr_sch.step()
    # torch.save(fcn.state_dict(), 'model_output/FCN_mnist_params.pkl')

    device = torch.device('cuda')
    sfcn = SpikingFCN(in_features=28 * 28 * 1, num_classes=10, if_param=IFParameters(), device=device,
                      seq_length=simulation_config['seq_length'],
                      max_firing_rate=simulation_config['max_firing_rate'],
                      dt=simulation_config['dt']).to(device)
    sfcn.load_state_dict(torch.load('model_output/FCN_mnist_params.pkl'))
    test_snn(model=sfcn, test_loader=test_loader, simulation_config=simulation_config)
