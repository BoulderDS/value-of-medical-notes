import argparse
import math
import time
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import DAN
import dataloader
from build_vocab import Vocabulary
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
import pandas as pd
def train_epoch(model, training_data, optimizer, loss_fn, device, opt):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss = 0
    count = 0
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        optimizer.zero_grad()
        # prepare data
        if opt.feature:
            note, length, mortality, feature = map(lambda x: x.to(device), batch)
            pred = model(note, length, feature)
        else:
            note, length, mortality = map(lambda x: x.to(device), batch)
            pred = model(note, length)
        # backward
        loss = loss_fn(pred, mortality.view(-1))
        loss.backward()
        optimizer.step()
        # note keeping
        total_loss += loss.item()

        count +=1
        #if count%10==0:
        #    print(f"Loss: {loss.item()}")
        #print("===============================================\n")

    loss = total_loss/count
    return loss

def eval_epoch(model, validation_data, loss_fn, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    count=0
    total_loss = 0
    true_all = []
    pred_all = []
    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):
            # prepare data
            if opt.feature:
                note, length, mortality, feature = map(lambda x: x.to(device), batch)
                pred = model(note, length, feature)
            else:
                note, length, mortality = map(lambda x: x.to(device), batch)
                pred = model(note, length)
            # backward
            loss = loss_fn(pred, mortality.view(-1))
            # note keeping
            total_loss += loss.item()
            count +=1
            # probability
            true_all.append(mortality.view(-1))
            pred_all.append(F.softmax(pred)[:,1].view(-1))
    true_all = torch.cat(true_all, axis=0)
    pred_all = torch.cat(pred_all, axis=0)
    roc_auc = roc_auc_score(true_all.cpu(), pred_all.cpu())
    precision, recall, thresholds = precision_recall_curve(true_all.cpu(), pred_all.cpu())
    pr_auc = auc(recall, precision)
    ap = average_precision_score(true_all.cpu(), pred_all.cpu())
    p_at_1 = precision_at_k(true_all.cpu(), pred_all.cpu(), 1)
    p_at_5 = precision_at_k(true_all.cpu(), pred_all.cpu(), 5)
    p_at_10 = precision_at_k(true_all.cpu(), pred_all.cpu(), 10)

    loss_per_word = total_loss/count
    print("ROC AUC:", roc_auc)
    print("PR AUC:", pr_auc)
    print("Loss:", loss_per_word)
    return loss_per_word, (roc_auc, pr_auc, ap, p_at_1, p_at_5, p_at_10)

def test(model, training_data, validation_data, test_data, loss_fn, device, opt):
    ''' Epoch operation in evaluation phase '''
    
    best_train_scores = eval_epoch(model, training_data, loss_fn, device, opt)[1]
    best_valid_scores = eval_epoch(model, validation_data, loss_fn, device, opt)[1]

    model.eval()
    count=0
    total_loss = 0
    true_all = []
    pred_all = []
    with torch.no_grad():
        for batch in tqdm(
                test_data, mininterval=2,
                desc='  - (Validation) ', leave=False):
            # prepare data
            if opt.feature:
                note, length, mortality, feature = map(lambda x: x.to(device), batch)
                pred = model(note, length, feature)
            else:
                note, length, mortality = map(lambda x: x.to(device), batch)
                pred = model(note, length)
            # backward
            loss = loss_fn(pred, mortality.view(-1))
            # note keeping
            total_loss += loss.item()
            count +=1
            # probability
            true_all.append(mortality.view(-1))
            pred_all.append(F.softmax(pred)[:,1].view(-1))
    true_all = torch.cat(true_all, axis=0)
    pred_all = torch.cat(pred_all, axis=0)
    roc_auc = roc_auc_score(true_all.cpu(), pred_all.cpu())
    precision, recall, thresholds = precision_recall_curve(true_all.cpu(), pred_all.cpu())
    pr_auc = auc(recall, precision)
    ap = average_precision_score(true_all.cpu(), pred_all.cpu())
    p_at_1 = precision_at_k(true_all.cpu(), pred_all.cpu(), 1)
    p_at_5 = precision_at_k(true_all.cpu(), pred_all.cpu(), 5)
    p_at_10 = precision_at_k(true_all.cpu(), pred_all.cpu(), 10)

    loss_per_word = total_loss/count
    print("----- Test Result -----")
    print("ROC AUC:", roc_auc)
    print("PR AUC:", pr_auc)
    print("Loss:", loss_per_word)
    if not os.path.exists("./results/"):
        os.mkdir("results")
    if not os.path.exists(f"./results/{opt.task}"):
        os.mkdir(f"./results/{opt.task}")
    if not os.path.exists(f"./results/{opt.task}/{opt.name}"):
        os.mkdir(f"./results/{opt.task}/{opt.name}")

    outname = f'{opt.period}.csv'
    if opt.text:
        outname = "text_" + outname
    if opt.feature:
        outname = "feature_" + outname

    print("Write Result to ", outname)
    with open(os.path.join('./results/', opt.task, opt.name, outname), 'w') as f:
        f.write("TYPE,ROCAUC,PRAUC,AP,P@1,P@5,P@10\n")
        f.write(f"train,{best_train_scores[0]},{best_train_scores[1]},{best_train_scores[2]},{best_train_scores[3]},{best_train_scores[4]},{best_train_scores[5]}\n")
        f.write(f"valid,{best_valid_scores[0]},{best_valid_scores[1]},{best_valid_scores[2]},{best_valid_scores[3]},{best_valid_scores[4]},{best_valid_scores[5]}\n")
        f.write(f"test,{roc_auc},{pr_auc},{ap},{p_at_1},{p_at_5},{p_at_10}")

def precision_at_k(y_label, y_pred, k):
    rank = list(zip(y_label, y_pred))
    rank.sort(key=lambda x: x[1], reverse=True)
    num_k = len(y_label)*k//100
    return sum(rank[i][0].item() == 1 for i in range(num_k))/float(num_k)

def train(model, training_data, validation_data, optimizer, loss_fn, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None
    log_dir = opt.log

    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        print(f"Scheduled Learning Rate:{lr}")
    if opt.log:
        log_name = opt.task+'_'+opt.name+'_'+opt.period
        if opt.text:
            log_name = "text_" + log_name
        if opt.feature:
            log_name = "feature_" + log_name
        log_train_file = log_dir + log_name + '.train.log'
        log_valid_file = log_dir + log_name + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            #log_tf.write('epoch,loss,ppl,accuracy\n')
            #log_vf.write('epoch,loss,ppl,accuracy\n')
            log_tf.write('epoch,loss,ppl\n')
            log_vf.write('epoch,loss,ppl\n')

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    valid_losses = []
    best = (0,0)
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss  = train_epoch(
            model, training_data, optimizer, loss_fn, device, opt=opt)
        #print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
        #      'elapse: {elapse:3.3f} min'.format(
        #          ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
        #          elapse=(time.time()-start)/60))
        print('  - (Training)   loss: {loss: 8.5f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_results= eval_epoch(model, validation_data, loss_fn, device, opt)
        print('  - (Validation) ppl: {ppl: 8.5f}, roc_auc: {accu:3.3f} %, pr_auc: {prauc:3.3f}'\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=valid_results[0], prauc=valid_results[2],
                    elapse=(time.time()-start)/60))
        #print('  - (Validation) loss: {loss: 8.5f}, '\
        #        'elapse: {elapse:3.3f} min'.format(
        #            loss=valid_loss,
        #            elapse=(time.time()-start)/60))
        scheduler.step(valid_results[2])
        valid_losses += [valid_results[2]]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            save_dir = f"{opt.data_dir}/Deep-Average-Network/models/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if opt.save_mode == 'all':
                model_name = save_dir + opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_loss)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':

                model_name = opt.task+'_'+opt.name +'_'+opt.period + '.chkpt'
                if opt.text:
                    model_name = "text_" + model_name
                if opt.feature:
                    model_name = "feature_" + model_name
                model_name = save_dir + model_name
                if valid_results[2] >= max(valid_losses):
                    best = valid_results
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                #log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                #    epoch=epoch_i, loss=train_loss,
                #    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f},{prauc}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=valid_results[0], prauc=valid_results[2]))
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100))))
                #log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f}\n'.format(
                #    epoch=epoch_i, loss=valid_loss,
                #    ppl=math.exp(min(valid_loss, 100))))
    return best

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=20)
    parser.add_argument('-batch_size', type=int, default=32)

    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-embedding_size', type=float, default=300)
    parser.add_argument('-learning_rate', type=float, default=0.0003)

    parser.add_argument('-name', type=str, default=None, choices=['all', 'all_but_discharge', 'physician', 'discharge', 'physician_nursing'])
    parser.add_argument('-task', type=str, default=None, choices=['mortality', 'readmission'])
    parser.add_argument('-data_name', type=str, default=None)
    parser.add_argument('-period', type=str, choices=['24', '48', 'retro'])
    parser.add_argument('-data_dir', type=str, required=True)

    parser.add_argument('-feature', action='store_true', default=False)
    parser.add_argument('-text', action='store_true', default=False)

    parser.add_argument('-compare_note', action='store_true', default=False)
    parser.add_argument('-text_length', action='store_true', default=False)
    parser.add_argument('-segment', type=str, default=None)


    parser.add_argument('-log', type=str, default="/data/joe/physician_notes/Deep-Average-Network/log/")
    parser.add_argument('-save_model', default=True)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-test_mode', action='store_true', default=False)

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-device', type=str, default='0')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.log = f"{opt.data_dir}/Deep-Average-Network/log/"
    if not os.path.exists(opt.log):
        os.mkdir(opt.log)
    #========= Loading Dataset =========#
    torch.manual_seed(1234)
    training_data, validation_data, test_data, vocab, feature_len = dataloader.get_loaders(opt, is_test=opt.test_mode, is_feature = opt.feature)

    #========= Preparing Model =========#
    print(opt)

    device = torch.device(f'cuda:{opt.device}' if opt.cuda else 'cpu')

    dan = DAN(len(vocab), opt.embedding_size, feature_len, opt.dropout, opt.feature, opt.text).to(device)
    optimizer = optim.AdamW(
            dan.parameters(),
            betas=(0.9, 0.98), eps=1e-09, lr=opt.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    valid_best_scores = None
    if not opt.test_mode:
        valid_best_scores = train(dan, training_data, validation_data, optimizer, loss_fn, device ,opt)

    model_name = opt.task+'_'+opt.name +'_'+ opt.period + '.chkpt'
    if opt.text:
        model_name = "text_" + model_name
    if opt.feature:
        model_name = "feature_" + model_name
    checkpoint = torch.load(f"{opt.data_dir}/Deep-Average-Network/models/{model_name}", map_location=device)
    dan.load_state_dict(checkpoint['model'])
    test(dan, training_data, validation_data, test_data, loss_fn, device, opt)
    #predict_prob(dan, test_data, loss_fn, device, opt)
if __name__ == '__main__':
    main()
