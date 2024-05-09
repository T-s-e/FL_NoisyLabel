import logging
import numpy as np
import copy

import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.losses import LogitAdjust, LA_KD, js

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    pred = np.array([])
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            pred = np.concatenate([pred, predicted.detach().cpu().numpy()], axis=0)
    return pred


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return self.idxs[item], image, label

    def get_num_of_each_class(self, args):
        class_sum = np.array([0] * args.n_classes)
        for idx in self.idxs:
            label = self.dataset.targets[idx]
            class_sum[label] += 1
        return class_sum.tolist()



class LocalUpdate(object):
    def __init__(self, args, id, dataset, idxs):
        self.args = args
        self.id = id
        self.idxs = idxs
        self.local_dataset = DatasetSplit(dataset, idxs)
        self.class_num_list = self.local_dataset.get_num_of_each_class(self.args)
        logging.info(
            f'client{id} each class num: {self.class_num_list}, total: {len(self.local_dataset)}')
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.epoch = 0
        self.round = 0
        self.iter_num = 0
        self.lr = self.args.base_lr


    def train_LA(self, net, writer):
        net.train()
        # set the optimizer
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        # train and update
        epoch_loss = []
        ce_criterion = LogitAdjust(cls_num_list=self.class_num_list)
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (_, images, labels) in self.ldr_train:
                images, labels = images.to(self.args.device), labels.cuda().to(self.args.device)

                logits = net(images)
                loss = ce_criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return net.state_dict(), np.array(epoch_loss).mean()
    

    def train_FedNoRo(self, student_net, teacher_net, writer, weight_kd):
        student_net.train()
        teacher_net.eval()
        # set the optimizer
        optimizer = torch.optim.Adam(
            student_net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        # train and update
        epoch_loss = []
        criterion = LA_KD(cls_num_list=self.class_num_list)
        
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                logits = student_net(images)
                with torch.no_grad():
                    teacher_output = teacher_net(images)
                    soft_label = torch.softmax(teacher_output/0.8, dim=1)

                loss = criterion(logits, labels, soft_label, weight_kd)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return student_net.state_dict(), np.array(epoch_loss).mean()

    def update_weights(self, net, seed, w_g, epoch, mu=1): # This function is the local training process of FedCorr
        net_glob = w_g

        net.train()
        # set the optimizer
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        logging.info(f"id: {self.id}, num: {len(self.local_dataset)}")

        self.loss_func = nn.CrossEntropyLoss()  # loss function -- cross entropy

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (_, images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                if self.args.mixup:
                    inputs, targets_a, targets_b, lam = mixup_data(images, labels, self.args.alpha)
                    net.zero_grad()
                    log_probs = net(inputs)
                    loss = mixup_criterion(self.loss_func, log_probs, targets_a, targets_b, lam)
                else:
                    labels = labels.long()
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)

                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_fedprox(self, net, writer):
        net.train()
        # set the optimizer
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        logging.info(f"id: {self.id}, num: {len(self.local_dataset)}")
        global_weight_collector = list(copy.deepcopy(net.cuda()).parameters())

        # train and update
        epoch_loss = []
        ce_criterion = nn.CrossEntropyLoss()
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (_, images, labels) in self.ldr_train:
                images, labels = images.cuda(), labels.cuda()

                logits = net(images)
                loss_ce = ce_criterion(logits, labels)

                # for fedprox
                fed_prox_reg = 0.0
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += (torch.norm((param - global_weight_collector[param_index]))) ** 2

                # loss = loss_ce + 0.01 * fed_prox_reg
                loss = loss_ce + self.args.prox * fed_prox_reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                writer.add_scalar(
                    f'client{self.id}/loss_ce', loss_ce, self.iter_num)
                writer.add_scalar(
                    f'client{self.id}/loss_fedprox', fed_prox_reg, self.iter_num)

                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return net.state_dict(), np.array(epoch_loss).mean()

    def train_FedLSR_plus(self, net, writer):
        net.train()
        # set the optimizer
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        logging.info(f"id: {self.id}, num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        ce_criterion = nn.CrossEntropyLoss()
        sm = torch.nn.Softmax(dim=1)
        lsm = torch.nn.LogSoftmax(dim=1)
        gamma = 0.4
        t_w = 0.2*self.args.rounds

        # data augmentation
        tt_transform = transforms.Compose([
            transforms.RandomRotation(30)])

        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (_, images, labels) in self.ldr_train:
                images, labels = images.cuda(), labels.cuda()

                images_aug = tt_transform(images)
                images_aug = images_aug.cuda()

                # output = net(images)
                # split_size = images.shape[0] // 2
                # output1, output2 = torch.split(output, [split_size, split_size], dim=0)

                output1 = net(images)
                output2 = net(images_aug)

                mix_1 = np.random.beta(1,1) # mixing predict1 and predict2
                mix_2 = 1 - mix_1

                logits1, logits2 = torch.softmax(output1*3, dim=1),torch.softmax(output2*3, dim=1)
                logits1, logits2 = torch.clamp(logits1, min=1e-6, max=1.0), torch.clamp(logits2, min=1e-6, max=1.0)

                L_e = - (torch.mean(torch.sum(sm(logits1) * lsm(logits1), dim=1)) +
                    torch.mean(torch.sum(sm(logits1) * lsm(logits1), dim=1))) * 0.5

                p = torch.softmax(output1, dim=1)*mix_1 + torch.softmax(output2, dim=1)*mix_2
                pt = p**(2)
                pred_mix = pt / pt.sum(dim=1, keepdim=True)


                betaa = gamma
                if self.round<t_w:
                    betaa = gamma * self.round / t_w

                loss = ce_criterion(pred_mix, labels)
                loss += js(logits1, logits2) * betaa
                loss += L_e * 0.6

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())
        self.round += 1
        return net.state_dict(), np.array(epoch_loss).mean()


    
class LocalUpdate_FedAvg(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.batch_size, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.base_lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (_, images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                # verbose print
                verbose = False

                if verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
