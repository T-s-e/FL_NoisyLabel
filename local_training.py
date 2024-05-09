import logging
import numpy as np
from tqdm import tqdm

import copy
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
from torchvision import transforms

from utils.losses import LogitAdjust, js, LA_smooth, LA_GLS, LA_GKD, SCELoss
from utils.utils import get_output


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
            if isinstance(outputs, tuple):
                outputs = outputs[0]
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

    def get_num_class_list(self, args):
        class_sum = np.array([0] * args.n_classes)
        for idx in self.idxs:
            label = self.dataset.targets[idx]
            class_sum[label] += 1
        return class_sum.tolist()

    def get_num_clean_class_list(self, args):
        class_sum = np.array([0] * args.n_classes)
        for idx in self.idxs:
            label = self.dataset.clean_targets[idx]
            class_sum[label] += 1
        return class_sum.tolist()


class LocalUpdate(object):
    def __init__(self, args, id, dataset, idxs):
        self.args = args
        self.id = id
        self.idxs = idxs
        self.local_dataset = DatasetSplit(dataset, idxs)
        self.class_num_list = self.local_dataset.get_num_class_list(self.args)
        logging.info(
            f'client{id} each class num: {self.class_num_list}, total: {len(self.local_dataset)}')
        logging.info(
            f'client{id} each class num (clean): {self.local_dataset.get_num_clean_class_list(self.args)}, total: {len(self.local_dataset)}')
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.epoch = 0
        self.round = 0
        self.iter_num = 0
        self.lr = self.args.base_lr


    def reset_data(self, dataset, idxs):
        assert set(idxs) == set(self.idxs)
        self.local_dataset = DatasetSplit(dataset, idxs)
        # self.class_num_list = self.local_dataset.get_num_class_list(self.args)
        # logging.info(
        #     f'client{self.id} each class num (after correcting): {self.class_num_list}, total: {len(self.local_dataset)}')
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
    

    def train(self, net, writer):
        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        ce_criterion = nn.CrossEntropyLoss()
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (_, images, labels) in self.ldr_train:
                images, labels = images.cuda(), labels.cuda()

                logits = net(images)
                loss = ce_criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return net.state_dict(), np.array(epoch_loss).mean()


    def train_fedprox(self, net, writer):
        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")
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

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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


    def train_FedLSR(self, net, writer):
        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        ce_criterion = nn.CrossEntropyLoss()
        sm = torch.nn.Softmax(dim=1)
        lsm = torch.nn.LogSoftmax(dim=1)
        gamma = 0.4
        t_w = 0.2*self.args.rounds
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (_, images, labels) in self.ldr_train:
                assert isinstance(images, list)
                images = torch.cat([images[0], images[1]], dim=0)
                images, labels = images.cuda(), labels.cuda()
                assert images.shape[0] == 2*labels.shape[0]
                
                output = net(images)
                split_size = images.shape[0] // 2
                output1, output2 = torch.split(output, [split_size, split_size], dim=0)

                # output1 = net(images)
                # output2 = net(images_aug)

                mix_1 = np.random.beta(1,1) # mixing predict1 and predict2
                mix_2 = 1 - mix_1

                logits1, logits2 = torch.softmax(output1*3, dim=1),torch.softmax(output2*3, dim=1)
                logits1, logits2 = torch.clamp(logits1, min=1e-6, max=1.0), torch.clamp(logits2, min=1e-6, max=1.0) 

                p = torch.softmax(output1, dim=1)*mix_1 + torch.softmax(output2, dim=1)*mix_2
                pt = p**(2)
                pred_mix = pt / pt.sum(dim=1, keepdim=True)

                betaa = gamma
                if self.round<t_w:
                    betaa = gamma * self.round / t_w

                loss = ce_criterion(pred_mix, labels)
                loss += js(logits1, logits2) * betaa

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())
        self.round += 1
        return net.state_dict(), np.array(epoch_loss).mean()


    def train_RoFLnew(self, net, writer):
        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        ce_criterion = SCELoss(alpha=0.1, beta=1.0, num_classes=self.args.n_classes)
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (_, images, labels) in self.ldr_train:
                images, labels = images.cuda(), labels.cuda()

                logits = net(images)
                loss = ce_criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return net.state_dict(), np.array(epoch_loss).mean()


    # def train_FedLSR_plus(self, net, writer):
    #     net.train()
    #     # set the optimizer
    #     self.optimizer = torch.optim.Adam(
    #         net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
    #     print(f"id: {self.id}, num: {len(self.local_dataset)}")

    #     # train and update
    #     epoch_loss = []
    #     ce_criterion = nn.CrossEntropyLoss()
    #     sm = torch.nn.Softmax(dim=1)
    #     lsm = torch.nn.LogSoftmax(dim=1)
    #     gamma = 0.4
    #     t_w = 0.2*self.args.rounds
    #     for epoch in range(self.args.local_ep):
    #         batch_loss = []
    #         for (images, labels) in self.ldr_train:
    #             assert isinstance(images, list)
    #             images = torch.cat([images[0], images[1]], dim=0)
    #             images, labels = images.cuda(), labels.cuda()
    #             assert images.shape[0] == 2*labels.shape[0]
                
    #             output = net(images)
    #             split_size = images.shape[0] // 2
    #             output1, output2 = torch.split(output, [split_size, split_size], dim=0)
                
    #             # output1 = net(images)
    #             # output2 = net(images_aug)

    #             mix_1 = np.random.beta(1,1) # mixing predict1 and predict2
    #             mix_2 = 1 - mix_1

    #             logits1, logits2 = torch.softmax(output1*3, dim=1),torch.softmax(output2*3, dim=1)
    #             logits1, logits2 = torch.clamp(logits1, min=1e-6, max=1.0), torch.clamp(logits2, min=1e-6, max=1.0) 

    #             L_e = - (torch.mean(torch.sum(sm(logits1) * lsm(logits1), dim=1)) +
    #                 torch.mean(torch.sum(sm(logits1) * lsm(logits1), dim=1))) * 0.5

    #             p = torch.softmax(output1, dim=1)*mix_1 + torch.softmax(output2, dim=1)*mix_2
    #             pt = p**(2)
    #             pred_mix = pt / pt.sum(dim=1, keepdim=True)

                
    #             betaa = gamma
    #             if self.round<t_w:
    #                 betaa = gamma * self.round / t_w

    #             loss = ce_criterion(pred_mix, labels)
    #             loss +=  js(logits1, logits2) * betaa
    #             loss += L_e * 0.6

    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()

    #             batch_loss.append(loss.item())
    #             writer.add_scalar(
    #                 f'client{self.id}/loss_train', loss.item(), self.iter_num)
    #             self.iter_num += 1
    #         self.epoch = self.epoch + 1
    #         epoch_loss.append(np.array(batch_loss).mean())
    #     self.round += 1
    #     return net.state_dict(), np.array(epoch_loss).mean()


    def train_LA(self, net, writer):
        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        ce_criterion = LogitAdjust(cls_num_list=self.class_num_list)
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                images, labels = images.cuda(), labels.cuda()

                logits = net(images)
                loss = ce_criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return net.state_dict(), np.array(epoch_loss).mean()


    # def train_LA_mixup(self, net, writer):
    #     net.train()
    #     # set the optimizer
    #     self.optimizer = torch.optim.Adam(
    #         net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
    #     print(f"id: {self.id}, num: {len(self.local_dataset)}")

    #     # train and update
    #     epoch_loss = []
    #     ce_criterion = LogitAdjust(cls_num_list=self.class_num_list)
    #     for epoch in range(self.args.local_ep):
    #         batch_loss = []
    #         for (images, labels) in self.ldr_train:
    #             images, labels = images.cuda(), labels.cuda()

    #             inputs, targets_a, targets_b, lam = mixup_data(images, labels, self.args.alpha)
    #             log_probs = net(inputs)
    #             loss = mixup_criterion(ce_criterion, log_probs, targets_a, targets_b, lam)

    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()

    #             batch_loss.append(loss.item())
    #             writer.add_scalar(
    #                 f'client{self.id}/loss_train', loss.item(), self.iter_num)
    #             self.iter_num += 1
    #         self.epoch = self.epoch + 1
    #         epoch_loss.append(np.array(batch_loss).mean())

    #     return net.state_dict(), np.array(epoch_loss).mean()


    # def train_LA_LS(self, net, writer):
    #     net.train()
    #     # set the optimizer
    #     self.optimizer = torch.optim.Adam(
    #         net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
    #     print(f"id: {self.id}, num: {len(self.local_dataset)}")

    #     # train and update
    #     epoch_loss = []
    #     ce_criterion = LA_smooth(cls_num_list=self.class_num_list)
    #     for epoch in range(self.args.local_ep):
    #         batch_loss = []
    #         for (_, images, labels) in self.ldr_train:
    #             images, labels = images.cuda(), labels.cuda()

    #             logits = net(images)
    #             loss = ce_criterion(logits, labels)

    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()

    #             batch_loss.append(loss.item())
    #             writer.add_scalar(
    #                 f'client{self.id}/loss_train', loss.item(), self.iter_num)
    #             self.iter_num += 1
    #         self.epoch = self.epoch + 1
    #         epoch_loss.append(np.array(batch_loss).mean())

    #     return net.state_dict(), np.array(epoch_loss).mean()


    def update_weights(self, net, seed, w_g, epoch, mu=1): # This function is the local training process of FedCorr
        net_glob = w_g

        net.train()
        # set the optimizer
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        self.loss_func = nn.CrossEntropyLoss()  # loss function -- cross entropy

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (_, images, labels) in enumerate(tqdm(self.ldr_train)):
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

    
    def update_weights_LA(self, net, seed, w_g, epoch, mu=1): # This function is the local training process of FedCorr
        net_glob = w_g

        net.train()
        # set the optimizer
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        self.loss_func = LogitAdjust(cls_num_list=self.class_num_list)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (_, images, labels) in enumerate(tqdm(self.ldr_train)):
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


    def train_0_1(self, net, writer):
        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        # self.easy_idxs = self.idxs
        # self.hard_idxs = []
        assert set(self.easy_idxs + self.hard_idxs) == set(self.idxs)

        # train and update
        epoch_loss = []
        criterion = LA_GLS(cls_num_list=self.class_num_list)
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                images, labels = images.cuda(), labels.cuda()

                logits = net(images)

                is_easy = [i in self.easy_idxs for i in img_idx]
                loss = criterion(logits, labels, is_easy)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return net.state_dict(), np.array(epoch_loss).mean()


    def train_0_2(self, net, glob_net_ema, writer):
        net.train()
        glob_net_ema.eval()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        criterion = LA_GKD(cls_num_list=self.class_num_list)
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                images, labels = images.cuda(), labels.cuda()

                logits = net(images)
                with torch.no_grad():
                    ema_output = glob_net_ema(images)
                    ema_pred = torch.softmax(ema_output, dim=1)

                loss = criterion(logits, labels, ema_pred)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return net.state_dict(), np.array(epoch_loss).mean()

    def train_dis(self, student_net, teacher_net, writer):
        student_net.train()
        teacher_net.eval()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            student_net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        criterion = LA_GKD(cls_num_list=self.class_num_list, eps=0.75)
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                images, labels = images.cuda(), labels.cuda()

                logits = student_net(images)
                with torch.no_grad():
                    ema_output = teacher_net(images)
                    ema_pred = torch.softmax(ema_output, dim=1)

                loss = criterion(logits, labels, ema_pred)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return student_net.state_dict(), np.array(epoch_loss).mean()

    
    def train_033(self, student_net, teacher_net, writer, weight_kd):
        student_net.train()
        teacher_net.eval()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            student_net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        criterion = LA_GKD(cls_num_list=self.class_num_list, eps=0.75)
        
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                images, labels = images.cuda(), labels.cuda()

                logits = student_net(images)
                with torch.no_grad():
                    ema_output = teacher_net(images)
                    ema_pred = torch.softmax(ema_output, dim=1)

                loss = criterion(logits, labels, ema_pred, weight_kd)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return student_net.state_dict(), np.array(epoch_loss).mean()

    
    def train_035(self, student_net, teacher_net, writer, weight_kd):
        student_net.train()
        teacher_net.eval()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            student_net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        criterion = LA_GKD(cls_num_list=self.class_num_list, eps=0.75)
        
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                images, labels = images.cuda(), labels.cuda()

                logits = student_net(images)
                with torch.no_grad():
                    ema_output = teacher_net(images)
                    ema_pred = torch.softmax(ema_output/0.8, dim=1)

                loss = criterion(logits, labels, ema_pred, weight_kd)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return student_net.state_dict(), np.array(epoch_loss).mean()

    
    def train_036(self, student_net, teacher_net, writer, weight_kd):
        student_net.train()
        teacher_net.eval()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            student_net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        criterion = LA_GKD(cls_num_list=self.class_num_list, eps=0.75)
        
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                assert isinstance(images, list)
                images0, images1 = images[0], images[1]
                images0, images1, labels = images0.cuda(), images1.cuda(), labels.cuda()
                split_size = labels.shape[0]
                assert images0.shape[0] == images1.shape[0] == split_size
                with torch.no_grad():
                    ema_output = teacher_net(images0)
                    ema_pred = torch.softmax(ema_output/0.8, dim=1)

                logits = student_net(torch.cat((images0, images1), dim=0))
                logits0, logits1 = torch.split(logits, [split_size, split_size], dim=0)

                loss1 = criterion(logits0, labels, ema_pred, weight_kd)
                loss_self = js(logits0, logits1)*weight_kd
                loss = loss1 + loss_self

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                writer.add_scalar(
                    f'client{self.id}/loss1_train', loss1.item(), self.iter_num)
                writer.add_scalar(
                    f'client{self.id}/lossself_train', loss_self.item(), self.iter_num)
                if self.iter_num % 100==0:
                    writer.add_image('train/input0', images0[0], self.iter_num)
                    writer.add_image('train/input1', images1[0], self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return student_net.state_dict(), np.array(epoch_loss).mean()

    
    def train_038(self, student_net, teacher_net, writer, weight_kd):
        student_net.train()
        teacher_net.eval()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            student_net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        # criterion = LA_GKD(cls_num_list=self.class_num_list, eps=0.75)
        criterion = LogitAdjust(cls_num_list=self.class_num_list)
        
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                assert isinstance(images, list)
                images0, images1 = images[0], images[1]
                images0, images1, labels = images0.cuda(), images1.cuda(), labels.cuda()
                split_size = labels.shape[0]
                assert images0.shape[0] == images1.shape[0] == split_size

                logits = student_net(torch.cat((images0, images1), dim=0))
                logits0, logits1 = torch.split(logits, [split_size, split_size], dim=0)

                loss1 = criterion(logits0, labels)
                loss_self = js(logits0, logits1)
                loss = loss1*(1-weight_kd) + loss_self*weight_kd

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                writer.add_scalar(
                    f'client{self.id}/loss1_train', loss1.item(), self.iter_num)
                writer.add_scalar(
                    f'client{self.id}/lossself_train', loss_self.item(), self.iter_num)
                if self.iter_num % 100==0:
                    writer.add_image('train/input0', images0[0], self.iter_num)
                    writer.add_image('train/input1', images1[0], self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return student_net.state_dict(), np.array(epoch_loss).mean()


    def train_0310(self, net, writer, weight_kd):
        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        ce_criterion = LogitAdjust(cls_num_list=self.class_num_list)
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                images, labels = images.cuda(), labels.cuda()

                logits = net(images)
                loss = ce_criterion(logits, labels) * (1-weight_kd)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return net.state_dict(), np.array(epoch_loss).mean()


    def train_0311(self, student_net, teacher_net, writer, weight_kd):
        student_net.train()
        teacher_net.eval()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            student_net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        criterion = LA_GKD(cls_num_list=self.class_num_list, eps=0.75)
        
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                assert isinstance(images, list)
                images0, images1 = images[0], images[1]
                images0, images1, labels = images0.cuda(), images1.cuda(), labels.cuda()
                split_size = labels.shape[0]
                assert images0.shape[0] == images1.shape[0] == split_size

                logits = student_net(torch.cat((images0, images1), dim=0))
                logits0, logits1 = torch.split(logits, [split_size, split_size], dim=0)
                ema_pred = torch.softmax(logits1/0.8, dim=1).detach()
                loss = criterion(logits0, labels, ema_pred, weight_kd)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                if self.iter_num % 100==0:
                    writer.add_image('train/input0', images0[0], self.iter_num)
                    writer.add_image('train/input1', images1[0], self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return student_net.state_dict(), np.array(epoch_loss).mean()

    
    def train_0312(self, student_net, teacher_net, writer, weight_kd):
        student_net.train()
        teacher_net.eval()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            student_net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"id: {self.id}, num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        criterion = LA_GKD(cls_num_list=self.class_num_list, eps=0.75)
        
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                assert isinstance(images, list)
                images0, images1 = images[0], images[1]
                images0, images1, labels = images0.cuda(), images1.cuda(), labels.cuda()
                split_size = labels.shape[0]
                assert images0.shape[0] == images1.shape[0] == split_size
                with torch.no_grad():
                    ema_output = teacher_net(images0)
                    ema_pred0 = torch.softmax(ema_output/0.8, dim=1)

                logits = student_net(torch.cat((images0, images1), dim=0))
                logits0, logits1 = torch.split(logits, [split_size, split_size], dim=0)
                ema_pred1 = torch.softmax(logits1/0.8, dim=1).detach()

                loss = criterion(logits0, labels, [ema_pred0, ema_pred1], weight_kd)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return student_net.state_dict(), np.array(epoch_loss).mean()





class DatasetSplitRFL(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label = self.dataset[self.idxs[item]]

        return image, label, self.idxs[item]
        

class LocalUpdateRFL:
    def __init__(self, args, dataset=None, user_idx=None, idxs=None):
        self.args = args
        self.dataset = dataset
        self.user_idx = user_idx
        self.idxs = idxs
        self.lr = self.args.base_lr
        
        self.pseudo_labels = torch.zeros(len(self.dataset), dtype=torch.long, device=self.args.device)
        self.sim = torch.nn.CosineSimilarity(dim=1) 
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        self.ldr_train = DataLoader(DatasetSplitRFL(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, num_workers=8, pin_memory=True)
        self.ldr_train_tmp = DataLoader(DatasetSplitRFL(dataset, idxs), batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
            
    def RFLloss(self, logit, labels, feature, f_k, mask, small_loss_idxs, new_labels):
        mse = torch.nn.MSELoss(reduction='none')
        ce = torch.nn.CrossEntropyLoss()
        sm = torch.nn.Softmax(dim=1)
        lsm = torch.nn.LogSoftmax(dim=1)
        
        L_c = ce(logit[small_loss_idxs], new_labels)
        L_cen = torch.sum(mask[small_loss_idxs] * torch.sum(mse(feature[small_loss_idxs], f_k[labels[small_loss_idxs]]), 1))
        L_e = -torch.mean(torch.sum(sm(logit[small_loss_idxs]) * lsm(logit[small_loss_idxs]), dim=1))
        
        lambda_e = self.args.lambda_e
        lambda_cen = self.args.lambda_cen
        if self.args.g_epoch < self.args.T_pl:
            lambda_cen = (self.args.lambda_cen * self.args.g_epoch) / self.args.T_pl
        
        return L_c + (lambda_cen * L_cen) + (lambda_e * L_e)
             
    def get_small_loss_samples(self, y_pred, y_true, forget_rate):
        loss = self.loss_func(y_pred, y_true)
        ind_sorted = np.argsort(loss.data.cpu()).cuda()
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update=ind_sorted[:num_remember]
        
        return ind_update
        
    def train_RoFL(self, net, f_G, writer):
        print(f"id: {self.user_idx}, num: {len(self.ldr_train.dataset)}")
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        epoch_loss = []
        net = net.cuda()
        import time
        
        t1 = time.time()
        net.eval()
        f_k = torch.zeros(self.args.num_classes, self.args.feature_dim, device=self.args.device)
        n_labels = torch.zeros(self.args.num_classes, 1, device=self.args.device)
        
        # obtain global-guided pseudo labels y_hat by y_hat_k = C_G(F_G(x_k))
        # initialization of global centroids
        # obtain naive average feature
        with torch.no_grad():
            for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train_tmp):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net(images)
                self.pseudo_labels[idxs] = torch.argmax(logit)    
                if self.args.g_epoch == 0:
                    f_k[labels] += feature
                    n_labels[labels] += 1
            
        if self.args.g_epoch == 0:
            for i in range(len(n_labels)):
                if n_labels[i] == 0:
                    n_labels[i] = 1           
            f_k = torch.div(f_k, n_labels)
        else:
            f_k = f_G

        t2 = time.time()

        net.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            correct_num = 0
            total = 0
            for batch_idx, batch in enumerate(self.ldr_train):
                net.zero_grad()        
                images, labels, idx = batch
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net(images)
                feature = feature.detach()
                f_k = f_k.to(self.args.device)
                
                small_loss_idxs = self.get_small_loss_samples(logit, labels, self.args.forget_rate)

                y_k_tilde = torch.zeros(self.args.local_bs, device=self.args.device)
                mask = torch.zeros(self.args.local_bs, device=self.args.device)
                for i in small_loss_idxs:
                    y_k_tilde[i] = torch.argmax(self.sim(f_k, torch.reshape(feature[i], (1, self.args.feature_dim))))
                    if y_k_tilde[i] == labels[i]:
                        mask[i] = 1
 
                # When to use pseudo-labels
                if self.args.g_epoch < self.args.T_pl:
                    for i in small_loss_idxs:    
                        self.pseudo_labels[idx[i]] = labels[i]
                
                # For loss calculating
                new_labels = mask[small_loss_idxs]*labels[small_loss_idxs] + (1-mask[small_loss_idxs])*self.pseudo_labels[idx[small_loss_idxs]]
                new_labels = new_labels.type(torch.LongTensor).to(self.args.device)
                
                loss = self.RFLloss(logit, labels, feature, f_k, mask, small_loss_idxs, new_labels)

                # weight update by minimizing loss: L_total = L_c + lambda_cen * L_cen + lambda_e * L_e
                loss.backward()
                optimizer.step()

                # obtain loss based average features f_k,j_hat from small loss dataset
                f_kj_hat = torch.zeros(self.args.num_classes, self.args.feature_dim, device=self.args.device)
                n = torch.zeros(self.args.num_classes, 1, device=self.args.device)
                for i in small_loss_idxs:
                    f_kj_hat[labels[i]] += feature[i]
                    n[labels[i]] += 1
                for i in range(len(n)):
                    if n[i] == 0:
                        n[i] = 1
                f_kj_hat = torch.div(f_kj_hat, n)

                # update local centroid f_k
                one = torch.ones(self.args.num_classes, 1, device=self.args.device)
                f_k = (one - self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_k + (self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_kj_hat

                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        t3 = time.time()

        # print(t3-t2, t2-t1)
        # input()
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), f_k