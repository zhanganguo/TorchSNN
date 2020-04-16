import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm


def test_snn(model, test_loader, simulation_config):
    model.eval()
    correct = 0
    total_accuracy_record = torch.zeros((simulation_config['seq_length'], 2))
    t_l =tqdm(test_loader)

    for data, target in t_l:
        if simulation_config['use_gpu']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        model.set_data_target(target)
        out, accuracy_record = model(data)
        total_accuracy_record += accuracy_record
        pred = out.data.max(1, keepdim=True)[1]
        cur_correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        correct += cur_correct
        t_l.set_description('correct: {} / {}'.format(cur_correct, len(data)))

    print('\nSpiking test set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    plt.title('{} Hz'.format(simulation_config['max_firing_rate']))
    plt.xlabel('time [ms]')
    plt.ylabel('accuracy [%]')
    x = np.arange(1, total_accuracy_record.shape[0] + 1)
    y = total_accuracy_record[:, 0] / total_accuracy_record[:, 1] * 100
    plt.plot(x, y)
    plt.show()