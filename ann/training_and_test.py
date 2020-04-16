from torch.autograd import Variable
from tqdm import tqdm


def train_ann(model, train_loader, optimizer, criterion, epoch, simulation_config):
    model.train()
    t_l = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(t_l):
        if simulation_config['use_gpu']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        out = model(data)

        loss = criterion(out, target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            t_l.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #    100. * batch_idx / len(train_loader), loss.data[0]))


def test_ann(model, test_loader, criterion, simulation_config):
    model.eval()
    test_loss = 0
    correct = 0
    t_l = tqdm(test_loader)
    for data, target in t_l:
        if simulation_config['use_gpu']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        out = model(data)
        test_loss += criterion(out, target.long()).item()
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
