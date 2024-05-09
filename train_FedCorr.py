import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
import sys
import copy
import time
import logging
import numpy as np
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

from utils.options import args_parser
from utils.local_training import LocalUpdate, globaltest
from utils.FedAvg import FedAvg
from utils.utils import add_attribute_noise, lid_term, get_output, set_seed, set_output_files

from dataset.dataset import get_dataset
from model.build_model import build_model

np.set_printoptions(threshold=np.inf)

"""
Major framework of noise FL
"""

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    args = args_parser()
    args.num_users = args.n_clients
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
    logging.info(f"train: {Counter(dataset_train.targets)}, total: {len(dataset_train.targets)}")
    logging.info(f"valid: {Counter(dataset_test.targets)}, total: {len(dataset_test.targets)}")

    # --------------------- Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_staining = np.array(dataset_train.staining)

    # specify the noise rate for each staining
    # staining_noise_rate = [0.4, 0.4, 0.3, 0.2, 0.1]
    staining_noise_rate = [0.7, 0.7, 0.5, 0.3, 0.1]
    logging.info(f"staining_noise_rate: {staining_noise_rate}")

    y_train_noisy, gamma_s, real_noise_level = add_attribute_noise(args, y_train, y_staining, dict_users, staining_noise_rate)
    dataset_train.clean_targets = np.array(dataset_train.targets)
    dataset_train.targets = y_train_noisy

    # --------------------- Build Models ---------------------------
    netglob = build_model(args)
    net_local = build_model(args)

    client_p_index = np.where(gamma_s == 0)[0]
    client_n_index = np.where(gamma_s > 0)[0]
    criterion = nn.CrossEntropyLoss(reduction='none')
    LID_accumulative_client = np.zeros(args.num_users)

    user_id = list(range(args.n_clients))
    trainer_locals = []
    for id in user_id:
        trainer_locals.append(LocalUpdate(
            args, id, copy.deepcopy(dataset_train), dict_users[id]))

    args.frac1 = 1 / args.n_clients
    # ------------------------------ begin training ------------------------------
    set_seed(args.seed)
    logging.info("\n ---------------------begin training---------------------")
    for iteration in range(args.iteration1):
        LID_whole = np.zeros(len(y_train))
        loss_whole = np.zeros(len(y_train))
        LID_client = np.zeros(args.num_users)
        loss_accumulative_whole = np.zeros(len(y_train))

        # ---------Broadcast global model----------------------
        if iteration == 0:
            mu_list = np.zeros(args.num_users)
        else:
            mu_list = estimated_noisy_level

        prob = [1 / args.num_users] * args.num_users

        for _ in range(int(1/args.frac1)):
            idxs_users = np.random.choice(range(args.num_users), int(args.num_users*args.frac1), p=prob)
            w_locals = []
            for idx in idxs_users:
                prob[idx] = 0
                if sum(prob) > 0:
                    prob = [prob[i] / sum(prob) for i in range(len(prob))]

                net_local.load_state_dict(netglob.state_dict())
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=32, shuffle=False)

                # proximal term operation
                mu_i = mu_list[idx]
                local = trainer_locals[idx]
                w, loss = local.update_weights(net=copy.deepcopy(net_local).to(args.device), seed=args.seed,
                                                w_g=netglob.to(args.device), epoch=args.local_ep, mu=mu_i)

                net_local.load_state_dict(copy.deepcopy(w))
                w_locals.append(copy.deepcopy(w))
                
                pred = globaltest(copy.deepcopy(net_local).to(args.device), dataset_test, args)
                acc = accuracy_score(dataset_test.targets, pred)
                bacc = balanced_accuracy_score(dataset_test.targets, pred)
                logging.info("iteration %d, client %d, acc: %.4f, bacc: %.4f" % (iteration, idx, acc, bacc))

                local_output, loss = get_output(loader, net_local.to(args.device), args, False, criterion)
                LID_local = list(lid_term(local_output, local_output))
                LID_whole[sample_idx] = LID_local
                loss_whole[sample_idx] = loss
                LID_client[idx] = np.mean(LID_local)

            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            w_glob = FedAvg(w_locals, dict_len)

            netglob.load_state_dict(copy.deepcopy(w_glob))

        LID_accumulative_client = LID_accumulative_client + np.array(LID_client)
        loss_accumulative_whole = loss_accumulative_whole + np.array(loss_whole)

        # Apply Gaussian Mixture Model to LID
        gmm_LID_accumulative = GaussianMixture(n_components=2, random_state=args.seed).fit(
            np.array(LID_accumulative_client).reshape(-1, 1))
        labels_LID_accumulative = gmm_LID_accumulative.predict(np.array(LID_accumulative_client).reshape(-1, 1))
        clean_label = np.argsort(gmm_LID_accumulative.means_[:, 0])[0]

        noisy_set = np.where(labels_LID_accumulative != clean_label)[0]
        clean_set = np.where(labels_LID_accumulative == clean_label)[0]

        logging.info(f"LID_accumulative: {LID_accumulative_client}")
        logging.info(f"noisy clients: {noisy_set}")

        estimated_noisy_level = np.zeros(args.num_users)

        for client_id in noisy_set:
            sample_idx = np.array(list(dict_users[client_id]))

            loss = np.array(loss_accumulative_whole[sample_idx])
            gmm_loss = GaussianMixture(n_components=2, random_state=args.seed).fit(np.array(loss).reshape(-1, 1))
            labels_loss = gmm_loss.predict(np.array(loss).reshape(-1, 1))
            gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]

            pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]
            estimated_noisy_level[client_id] = len(pred_n) / len(sample_idx)
            y_train_noisy_new = np.array(dataset_train.targets)

        if args.correction:
            for idx in noisy_set:
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                loss = np.array(loss_accumulative_whole[sample_idx])
                local_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
                # local_output = torch.softmax(torch.tensor(local_output), dim=-1).numpy()
                relabel_idx = (-loss).argsort()[:int(len(sample_idx) * estimated_noisy_level[idx] * args.relabel_ratio)]
                relabel_idx = list(set(np.where(np.max(local_output, axis=1) > args.confidence_thres)[0]) & set(relabel_idx))
                y_train_noisy_new = np.array(dataset_train.targets)
                nrb = (y_train_noisy_new[sample_idx] != dataset_train.clean_targets[sample_idx]).mean()
                y_train_noisy_new[sample_idx[relabel_idx]] = np.argmax(local_output, axis=1)[relabel_idx]
                nra = (y_train_noisy_new[sample_idx] != dataset_train.clean_targets[sample_idx]).mean()
                dataset_train.targets = y_train_noisy_new
                logging.info(f"client: {idx}, noise rate before: {nrb}, noise rate after: {nra}")
    logging.info('\n')

    # reset the beta,
    args.beta = 0

    # ---------------------------- second stage training -------------------------------
    BACC = []
    if args.fine_tuning:
        selected_clean_idx = np.where(estimated_noisy_level <= args.clean_set_thres)[0]
    
        prob = np.zeros(args.num_users)
        prob[selected_clean_idx] = 1 / len(selected_clean_idx)
        m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
        m = min(m, len(selected_clean_idx))
        netglob = copy.deepcopy(netglob)
        # add fl training
        for rnd in range(args.rounds1):
            w_locals, loss_locals = [], []
            idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
            for idx in idxs_users:  # training over the subset
                local = trainer_locals[idx]
                w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
                                                           w_g=netglob.to(args.device), epoch=args.local_ep, mu=0)
                w_locals.append(copy.deepcopy(w_local))  # store every updated model
                loss_locals.append(copy.deepcopy(loss_local))

            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            w_glob_fl = FedAvg(w_locals, dict_len)
            netglob.load_state_dict(copy.deepcopy(w_glob_fl))
    
            pred = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
            acc = accuracy_score(dataset_test.targets, pred)
            bacc = balanced_accuracy_score(dataset_test.targets, pred)
            cm = confusion_matrix(dataset_test.targets, pred)
            logging.info("fine tuning stage round %d, acc: %.4f, bacc: %.4f \n" % (rnd, acc, bacc))
            logging.info(cm)
            writer.add_scalar(f'test/acc', acc, rnd)
            writer.add_scalar(f'test/bacc', bacc, rnd)
            BACC.append(bacc)

        if args.correction:
            relabel_idx_whole = []
            for idx in noisy_set:
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                glob_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
                # glob_output = torch.softmax(torch.tensor(glob_output), dim=-1).numpy()
                y_predicted = np.argmax(glob_output, axis=1)
                relabel_idx = np.where(np.max(glob_output, axis=1) > args.confidence_thres)[0]
                y_train_noisy_new = np.array(dataset_train.targets)
                nrb = (y_train_noisy_new[sample_idx] != dataset_train.clean_targets[sample_idx]).mean()
                y_train_noisy_new[sample_idx[relabel_idx]] = y_predicted[relabel_idx]
                nra = (y_train_noisy_new[sample_idx] != dataset_train.clean_targets[sample_idx]).mean()
                dataset_train.targets = y_train_noisy_new
                logging.info(f"client: {idx}, noise rate before: {nrb}, noise rate after: {nra}")

    # ---------------------------- third stage training -------------------------------
    # third stage hyper-parameter initialization
    # m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
    # prob = [1/args.num_users for i in range(args.num_users)]
    
    best_performance = 0.
    for rnd in range(args.rounds2):
        w_locals, loss_locals = [], []
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in user_id:  # training over the subset
            local = trainer_locals[idx]
            w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
                                                        w_g=netglob.to(args.device), epoch=args.local_ep, mu=0)
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(copy.deepcopy(loss_local))

        dict_len = [len(dict_users[idx]) for idx in user_id]
        w_glob_fl = FedAvg(w_locals, dict_len)
        netglob.load_state_dict(copy.deepcopy(w_glob_fl))

        pred = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
        acc = accuracy_score(dataset_test.targets, pred)
        bacc = balanced_accuracy_score(dataset_test.targets, pred)
        cm = confusion_matrix(dataset_test.targets, pred)
        logging.info("******** third stage round %d, acc: %.4f, bacc: %.4f ********" % (rnd, acc, bacc))
        logging.info(cm)
        writer.add_scalar(f'test/acc', acc, rnd+args.rounds1)
        writer.add_scalar(f'test/bacc', bacc, rnd+args.rounds1)
        BACC.append(bacc)

        # save model
        if bacc > best_performance:
            best_performance = bacc
            # torch.save(netglob.state_dict(),  models_dir +
            #            f'/best_model_{rnd}_{best_performance}.pth')
            # torch.save(netglob.state_dict(),  models_dir+'/best_model.pth')
        logging.info(f'best bacc: {best_performance}, now bacc: {bacc}')

        logging.info('\n')

    BACC = np.array(BACC)
    logging.info("last:")
    logging.info(BACC[-10:].mean())
    logging.info("best:")
    logging.info(BACC.max())

    torch.cuda.empty_cache()