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
import Constants
from build_vocab import Vocabulary
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd

def test(model, test_data, loss_fn, device, opt):
    ''' Epoch operation in evaluation phase '''

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
            #print(pred)
    true_all = torch.cat(true_all, axis=0)
    pred_all = torch.cat(pred_all, axis=0)
    #print(true_all, pred_all)
    roc_auc = roc_auc_score(true_all.cpu(), pred_all.cpu())
    pr_auc = average_precision_score(true_all.cpu(), pred_all.cpu())
    loss_per_word = total_loss/count

    return pred_all.cpu()


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
    parser.add_argument('-segment',  type=str, default=None)
    parser.add_argument('-text_length', type=int, help='text length', default=None)

    parser.add_argument('-feature', action='store_true', default=False)
    parser.add_argument('-text', action='store_true', default=False)

    parser.add_argument('-log', type=str, default="/data/joe/physician_notes/Deep-Average-Network/log/")
    parser.add_argument('-save_model', default=True)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-test_mode', action='store_true', default=False)

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-device', type=str, default='0')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.compare_note = None
    #========= Loading Dataset =========#
    torch.manual_seed(1234)
    training_data, validation_data, test_data, vocab, feature_len = dataloader.get_loaders(opt, is_test=opt.test_mode, is_feature = opt.feature)

    #========= Preparing Model =========#
    print(opt)

    device = torch.device(f'cuda:{opt.device}' if opt.cuda else 'cpu')
    dan = DAN(len(vocab), opt.embedding_size, feature_len, opt.dropout, opt.feature, opt.text).to(device)
    loss_fn = nn.CrossEntropyLoss()

    model_name = opt.task+'_'+opt.name +'_'+ opt.period + '.chkpt'
    if opt.text:
        model_name = "text_" + model_name
    if opt.feature:
        model_name = "feature_" + model_name
    checkpoint = torch.load(f"/data/joe/physician_notes/Deep-Average-Network/models/{model_name}", map_location=device)
    dan.load_state_dict(checkpoint['model'])

    results = {}
    for note_id in Constants.note_type[opt.name]:
        opt.compare_note = note_id
        training_data, validation_data, test_data, vocab, feature_len = dataloader.get_loaders(opt, is_test=opt.test_mode, is_feature = opt.feature)
        res = test(dan, test_data, loss_fn, device, opt)
        results[note_id] = res
    #predict_prob(dan, test_data, loss_fn, device, opt)

    TEST_NOTE_PATH = f"/data/joe/physician_notes/mimic-data/{opt.task}/{opt.name}_note_test_{opt.period}.csv"
    test_file = pd.read_csv(TEST_NOTE_PATH)
    df = pd.DataFrame(results)
    df.insert(0,'stay', test_file['stay'])
    if not os.path.exists('/home/joe/physician_notes/models/DeepAverageNetwork/compare_notes/'):
        os.mkdir('/home/joe/physician_notes/models/DeepAverageNetwork/compare_notes/')
    model_name = opt.task+'_'+opt.name +'_'+ opt.period + '.csv'
    if opt.text:
        model_name = "text_" + model_name
    if opt.feature:
        model_name = "feature_" + model_name
    if opt.segment:
        model_name = opt.segment+ "_" + model_name
    df.to_csv(f'/home/joe/physician_notes/models/DeepAverageNetwork/compare_notes/{model_name}', index=False)

if __name__ == '__main__':
    main()
