import logging
import numpy as np

import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils.losses import LogitAdjust, LA_KD


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
                verbose = True

                if verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
