import os
import copy
import logging
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from collections import Counter
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from utils.options import args_parser
from utils.local_training import LocalUpdate_FedAvg, globaltest
from utils.FedAvg import FedAvg, DaAgg
from utils.utils import add_noise, add_attribute_noise, set_seed, set_output_files, get_output, get_current_consistency_weight

from dataset.dataset import get_dataset
from model.build_model import build_model
from model.Nets import CNN
from model.FedAvg import FedAvg, test_img

np.set_printoptions(threshold=np.inf)

"""
Major framework of noise FL
"""

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    args = args_parser()
    args.num_users = args.n_clients     # default: 20
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed)

    # ------------------------------ output files ------------------------------
    writer, models_dir = set_output_files(ROOT_DIR, args)

    # ------------------------------ dataset ------------------------------
    dataset_train, dataset_test, dict_users = get_dataset(args)
    logging.info(
        f"train: {Counter(dataset_train.targets)}, total: {len(dataset_train.targets)}")
    logging.info(
        f"test: {Counter(dataset_test.targets)}, total: {len(dataset_test.targets)}")

    # --------------------- Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_staining = np.array(dataset_train.staining)

    # specify the noise rate for each staining
    # staining_noise_rate = [0.4, 0.4, 0.3, 0.2, 0.1]
    staining_noise_rate = [0.7, 0.7, 0.5, 0.3, 0.1]
    logging.info(f"staining_noise_rate: {staining_noise_rate}")

    y_train_noisy, gamma_s, real_noise_level = add_attribute_noise(
        args, y_train, y_staining, dict_users, staining_noise_rate)
    dataset_train.targets = y_train_noisy

    # --------------------- Build Models ---------------------------

    img_size = dataset_train[0][0].shape

    net_glob = CNN(args=args).to(args.device)
    logging.info(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # --------------------- Training ---------------------------
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    w_locals = [w_glob for _ in range(args.n_clients)]
    for iter in range(args.rounds):
        loss_locals = []

        # All clients update
        idxs_users = np.array(range(args.n_clients))

        for idx in idxs_users:
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_glob[idx] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
            # print client loss
            logging.info(f'Round {iter}, User {idx}, loss: {loss:.3f}')

        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        logging.info('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # Save the model
    torch.save(net_glob.state_dict(), models_dir + '/final_model.pth')

    # plot loss curve
    plt.figure()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    plt.plot(range(len(loss_train)), loss_train, label='train')
    plt.plot(range(len(loss_test)), loss_test, label='test')
    plt.title('Camelyon17 CNN')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f'{models_dir}/loss.png')
    plt.close()

    logging.info("Training accuracy: {:.2f}".format(acc_train))
    logging.info("Testing accuracy: {:.2f}".format(acc_test))

    torch.cuda.empty_cache()







    # netglob = build_model(args)
    # user_id = list(range(args.n_clients))
    # trainer_locals = []
    # for id in user_id:
    #     trainer_locals.append(LocalUpdate(
    #         args, id, copy.deepcopy(dataset_train), dict_users[id]))
    #
    # # ------------------------------ begin training ------------------------------
    # set_seed(args.seed)
    # logging.info("\n ---------------------begin training---------------------")
    # best_performance = 0.
    #
    # # ------------------------ Stage 1: warm up ------------------------
    # logging.info("\n ---------------------Stage 1: warm up---------------------")
    # if args.warm:
    #     for rnd in range(args.s1):
    #         w_locals, loss_locals = [], []
    #         for idx in user_id:  # training over the subset
    #             local = trainer_locals[idx]
    #             w_local, loss_local = local.train_LA(
    #                 net=copy.deepcopy(netglob).to(args.device), writer=writer)
    #
    #             # store every updated model
    #             w_locals.append(copy.deepcopy(w_local))
    #             loss_locals.append(copy.deepcopy(loss_local))
    #             print(f"client {idx} loss: {loss_local}")
    #
    #         w_locals_last = copy.deepcopy(w_locals)
    #         dict_len = [len(dict_users[idx]) for idx in user_id]
    #         w_glob_fl = FedAvg(w_locals, dict_len)
    #         netglob.load_state_dict(copy.deepcopy(w_glob_fl))
    #
    #         pred = globaltest(copy.deepcopy(netglob).to(
    #             args.device), dataset_test, args)
    #         acc = accuracy_score(dataset_test.targets, pred)
    #         bacc = balanced_accuracy_score(dataset_test.targets, pred)
    #         cm = confusion_matrix(dataset_test.targets, pred)
    #         logging.info(
    #             "******** round: %d, acc: %.4f, bacc: %.4f ********" % (rnd, acc, bacc))
    #         logging.info(cm)
    #         writer.add_scalar(f'test/acc', acc, rnd)
    #         writer.add_scalar(f'test/bacc', bacc, rnd)
    #
    #         # save model
    #         if bacc > best_performance:
    #             best_performance = bacc
    #         logging.info(f'best bacc: {best_performance}, now bacc: {bacc}')
    #         logging.info('\n')
    #     torch.save(netglob.state_dict(),  models_dir +
    #                f'/stage1_model_{rnd}.pth')
    #
    # #  ------------------------ client selection ------------------------
    # logging.info("\n ---------------------client selection---------------------")
    # model_path = f"{ROOT_DIR}/outputs_{args.dataset}_{args.level_n_system}_{args.level_n_lowerb}_{args.level_n_upperb}/{args.exp}_{args.level_n_system}_{args.level_n_lowerb}_{args.level_n_upperb}_{args.local_ep}/models/stage1_model_{args.s1-1}.pth"
    # logging.info(
    #     f"********************** load model from: {model_path} **********************")
    # netglob.load_state_dict(torch.load(model_path))
    # loader = DataLoader(dataset=dataset_train, batch_size=32,
    #                     shuffle=False, num_workers=0)
    # criterion = nn.CrossEntropyLoss(reduction='none')
    # local_output, loss = get_output(
    #     loader, netglob.to(args.device), args, False, criterion)
    # logging.info(f"local_output: {local_output.shape}, loss: {loss.shape}")
    #
    # metrics = np.zeros((args.n_clients, args.n_classes)).astype("float")
    # num = np.zeros((args.n_clients, args.n_classes)).astype("float")
    # for id in range(args.n_clients):
    #     idxs = dict_users[id]
    #     for idx in idxs:
    #         c = dataset_train.targets[idx]
    #         num[id, c] += 1                     # per class number in each client
    #         metrics[id, c] += loss[idx]         # per class loss in each client
    # metrics = metrics / num                     # average loss per class in each client
    #
    # # process nan
    # for i in range(metrics.shape[0]):
    #     for j in range(metrics.shape[1]):
    #         if np.isnan(metrics[i, j]):
    #             metrics[i, j] = np.nanmin(metrics[:, j])
    # # normalize
    # for j in range(metrics.shape[1]):
    #     metrics[:, j] = (metrics[:, j]-metrics[:, j].min()) / \
    #         (metrics[:, j].max()-metrics[:, j].min())
    #
    # logging.info("metrics:")
    # logging.info(metrics)
    #
    # vote = []
    # for i in range(9):
    #     gmm = GaussianMixture(n_components=2, random_state=i).fit(metrics)
    #     gmm_pred = gmm.predict(metrics)
    #     noisy_clients = np.where(gmm_pred == np.argmax(gmm.means_.sum(1)))[0]
    #     noisy_clients = set(list(noisy_clients))
    #     vote.append(noisy_clients)
    # cnt = []
    # for i in vote:
    #     cnt.append(vote.count(i))
    # noisy_clients = list(vote[cnt.index(max(cnt))])
    #
    # logging.info(
    #     f"selected noisy clients: {noisy_clients}, real noisy clients: {np.where(gamma_s>0.)[0]}")
    # clean_clients = list(set(user_id) - set(noisy_clients))
    # logging.info(f"selected clean clients: {clean_clients}")
    #
    # # ------------------------ Stage 2: ------------------------
    # logging.info("\n ---------------------Stage 2---------------------")
    # BACC = []
    # for rnd in range(args.s1, args.rounds):             # default: 100 rnds
    #     w_locals, loss_locals = [], []
    #     weight_kd = get_current_consistency_weight(
    #         rnd, args.begin, args.end) * args.a
    #     writer.add_scalar(f'train/w_kd', weight_kd, rnd)
    #     for idx in user_id:  # training over the subset
    #         local = trainer_locals[idx]
    #         if idx in clean_clients:
    #             logging.info('clean clients')
    #             w_local, loss_local = local.train_LA(
    #                 net=copy.deepcopy(netglob).to(args.device), writer=writer)
    #         elif idx in noisy_clients:
    #             logging.info('noisy clients')
    #             w_local, loss_local = local.train_FedNoRo(
    #                 student_net=copy.deepcopy(netglob).to(args.device), teacher_net=copy.deepcopy(netglob).to(args.device), writer=writer, weight_kd=weight_kd)
    #         # store every updated model
    #         w_locals.append(copy.deepcopy(w_local))
    #         loss_locals.append(copy.deepcopy(loss_local))
    #         assert len(w_locals) == len(loss_locals) == idx+1
    #         logging.info(f"Round {rnd}: client {idx} loss: {loss_local}")
    #
    #     dict_len = [len(dict_users[idx]) for idx in user_id]
    #     w_glob_fl = DaAgg(
    #         w_locals, dict_len, clean_clients, noisy_clients)
    #     netglob.load_state_dict(copy.deepcopy(w_glob_fl))
    #
    #     pred = globaltest(copy.deepcopy(netglob).to(
    #         args.device), dataset_test, args)
    #     acc = accuracy_score(dataset_test.targets, pred)
    #     bacc = balanced_accuracy_score(dataset_test.targets, pred)
    #     cm = confusion_matrix(dataset_test.targets, pred)
    #     logging.info(
    #         "******** round: %d, acc: %.4f, bacc: %.4f ********" % (rnd, acc, bacc))
    #     logging.info(cm)
    #     writer.add_scalar(f'test/acc', acc, rnd)
    #     writer.add_scalar(f'test/bacc', bacc, rnd)
    #     BACC.append(bacc)
    #
    #     # save model
    #     if bacc > best_performance:
    #         best_performance = bacc
    #     logging.info(f'best bacc: {best_performance}, now bacc: {bacc}')
    #     logging.info('\n')
    # torch.save(netglob.state_dict(),  models_dir+f'/stage2_model_{rnd}.pth')
    #
    # BACC = np.array(BACC)
    # logging.info("last:")
    # logging.info(BACC[-10:].mean())
    # logging.info("best:")
    # logging.info(BACC.max())
    #
    # torch.cuda.empty_cache()