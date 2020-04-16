from typing import NamedTuple

import torch

from snn.utils import threshold


class AbstractSpikingNeuron:
    def __init__(self):
        pass

    def initialize_state(self, *kwargs):
        pass

    def forward(self, *kwargs):
        pass


class IFParameters(NamedTuple):
    """Parametrization of a LIF neuron

    Parameters:
        tau_mem (torch.Tensor): membrane time constant)
        resistance_mem (torch.Tensor): synaptic time constant)
        v_th (torch.Tensor): threshhold potential
        v_reset (torch.Tensor): reset potential
    """
    tau_mem: torch.Tensor = torch.as_tensor(1.0)
    resistance_mem: torch.Tensor = torch.as_tensor(1.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    t_refrac: torch.Tensor = torch.as_tensor(0.0)


class IfNeuron(AbstractSpikingNeuron):
    def __init__(self, shape, if_params: IFParameters):
        super(AbstractSpikingNeuron, self).__init__()
        self.shape = shape
        self.if_parameters = if_params

        self.v_mem = None

    def initialize_state(self, batch_size, device):
        self.v_mem = torch.zeros(batch_size, *self.shape, device=device)

    def forward(self, input_current):
        self.v_mem += input_current
        spike = threshold(self.v_mem, self.if_parameters.v_th)
        self.v_mem = (1 - spike) * self.v_mem + spike * self.if_parameters.v_reset
        return spike
