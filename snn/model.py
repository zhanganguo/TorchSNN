import torch
from ann.model import LeNet, FCN, Cifar10Net
from snn.spiking_operation import SpikingFeedForwardCell
from snn.spiking_neuron import IfNeuron
from snn.utils import threshold


class SpikingLeNet(LeNet):
    def __init__(self, input_shape, num_classes, if_param, device, seq_length, max_firing_rate, dt):
        super(SpikingLeNet, self).__init__(input_shape=input_shape, num_classes=num_classes)

        self.spiking_conv1 = SpikingFeedForwardCell(spiking_op=self.conv1, spiking_neuron=IfNeuron(
            shape=(16, self.input_shape[0] - 4, self.input_shape[1] - 4), if_params=if_param))

        self.spiking_conv2 = SpikingFeedForwardCell(spiking_op=self.conv2, spiking_neuron=IfNeuron(
            shape=(64, (self.input_shape[0] - 4) // 2 - 4, (self.input_shape[1] - 4) // 2 - 4), if_params=if_param))

        self.spiking_linear1 = SpikingFeedForwardCell(spiking_op=self.linear1, spiking_neuron=IfNeuron(shape=(128,), if_params=if_param))
        self.spiking_linear2 = SpikingFeedForwardCell(spiking_op=self.linear2, spiking_neuron=IfNeuron(shape=(num_classes,), if_params=if_param))

        self.seq_length = seq_length
        self.max_firing_rate = max_firing_rate
        self.dt = dt
        self.device = device

    def set_data_target(self, y):
        self.y = y

    def forward(self, x):
        batch_size = x.shape[0]
        self.spiking_conv1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_conv2.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear2.initial_state(batch_size=batch_size, device=self.device)
        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        accuracy_record = torch.zeros((self.seq_length, 2), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            inp_image = threshold(x, spike_snapshot)

            x_spike = self.spiking_conv1.forward(inp_image)
            x_spike = torch.nn.functional.avg_pool2d(x_spike, kernel_size=(2, 2), stride=(2, 2))
            x_spike = self.spiking_conv2.forward(x_spike)
            x_spike = torch.nn.functional.avg_pool2d(x_spike, kernel_size=(2, 2), stride=(2, 2))
            x_spike = x_spike.view(-1, self.num_flat_features(x_spike))
            x_spike = self.spiking_linear1.forward(x_spike)
            x_spike = self.spiking_linear2.forward(x_spike)

            spike_out += x_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            cur_correct = pred.eq(self.y.data.view_as(pred)).cpu().sum()
            accuracy_record[i, 0] = cur_correct
            accuracy_record[i, 1] = batch_size

        return spike_out, accuracy_record


class SpikingFCN(FCN):
    def __init__(self, in_features, num_classes, if_param, device, seq_length, max_firing_rate, dt):
        super(SpikingFCN, self).__init__(in_features, num_classes)

        self.spiking_linear1 = SpikingFeedForwardCell(spiking_op=self.linear1, spiking_neuron=IfNeuron(shape=(800,),  if_params=if_param))
        self.spiking_linear2 = SpikingFeedForwardCell(spiking_op=self.linear2, spiking_neuron=IfNeuron(shape=(800,),  if_params=if_param))
        self.spiking_linear3 = SpikingFeedForwardCell(spiking_op=self.linear3, spiking_neuron=IfNeuron(shape=(num_classes,),  if_params=if_param))

        self.seq_length = seq_length
        self.max_firing_rate = max_firing_rate
        self.dt = dt
        self.device = device

    def set_data_target(self, y):
        self.y = y

    def forward(self, x):
        batch_size = x.shape[0]
        self.spiking_linear1.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear2.initial_state(batch_size=batch_size, device=self.device)
        self.spiking_linear3.initial_state(batch_size=batch_size, device=self.device)

        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        accuracy_record = torch.zeros((self.seq_length, 2), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            x_spike = threshold(x, spike_snapshot)

            x_spike = self.spiking_linear1.forward(x_spike)
            x_spike = self.spiking_linear2.forward(x_spike)
            x_spike = self.spiking_linear3.forward(x_spike)

            spike_out += x_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            cur_correct = pred.eq(self.y.data.view_as(pred)).cpu().sum()
            accuracy_record[i, 0] = cur_correct
            accuracy_record[i, 1] = batch_size

        return spike_out, accuracy_record


class SpikingCifar10Net(Cifar10Net):
    def __init__(self, input_shape, num_classes, if_param, device, seq_length, max_firing_rate, dt):
        super(SpikingCifar10Net, self).__init__(input_shape=input_shape, num_classes=num_classes)
        self.spiking_conv1 = SpikingFeedForwardCell(spiking_op=self.conv1, spiking_neuron=IfNeuron(
            shape=(32, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv2 = SpikingFeedForwardCell(spiking_op=self.conv2, spiking_neuron=IfNeuron(
            shape=(64, self.input_shape[0], self.input_shape[1]), if_params=if_param))
        self.spiking_conv3 = SpikingFeedForwardCell(spiking_op=self.conv3, spiking_neuron=IfNeuron(
            shape=(128, self.input_shape[0]//2, self.input_shape[1]//2), if_params=if_param))
        self.spiking_conv4 = SpikingFeedForwardCell(spiking_op=self.conv4, spiking_neuron=IfNeuron(
            shape=(128, self.input_shape[0]//4, self.input_shape[1]//4), if_params=if_param))
        self.spiking_linear1 = SpikingFeedForwardCell(spiking_op=self.linear1, spiking_neuron=IfNeuron(shape=(256,), if_params=if_param))
        self.spiking_linear2 = SpikingFeedForwardCell(spiking_op=self.linear2, spiking_neuron=IfNeuron(shape=(num_classes,), if_params=if_param))

        self.seq_length = seq_length
        self.max_firing_rate = max_firing_rate
        self.dt = dt
        self.device = device

    def set_data_target(self, y):
        self.y = y

    def forward(self, x):
        batch_size = x.shape[0]
        self.if_conv1.initial_state(batch_size=batch_size, device=self.device)
        self.if_conv2.initial_state(batch_size=batch_size, device=self.device)
        self.if_conv3.initial_state(batch_size=batch_size, device=self.device)
        self.if_conv4.initial_state(batch_size=batch_size, device=self.device)
        self.if_linear1.initial_state(batch_size=batch_size, device=self.device)
        self.if_linear2.initial_state(batch_size=batch_size, device=self.device)
        spike_out = torch.zeros((batch_size, self.num_classes), device=self.device)
        accuracy_record = torch.zeros((self.seq_length, 2), device='cpu')

        for i in range(self.seq_length):
            rescale_fac = 1 / (self.max_firing_rate * self.dt) * (torch.max(x) - torch.min(x))
            spike_snapshot = rescale_fac * torch.rand_like(x, device=self.device)
            inp_image = threshold(x, spike_snapshot)

            x_spike = self.spiking_conv1.forward(inp_image)
            x_spike = self.spiking_conv2.forward(x_spike)
            x_spike = self.spiking_conv3.forward(x_spike)
            x_spike = self.spiking_conv4.forward(x_spike)
            x_spike = x_spike.view(-1, self.num_flat_features(x_spike))
            x_spike = self.spiking_linear1.forward(x_spike)
            x_spike = self.spiking_linear2.forward(x_spike)

            spike_out += x_spike

            pred = spike_out.data.max(1, keepdim=True)[1]
            cur_correct = pred.eq(self.y.data.view_as(pred)).cpu().sum()
            accuracy_record[i, 0] = cur_correct
            accuracy_record[i, 1] = batch_size

        return spike_out, accuracy_record
