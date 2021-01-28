
import numpy as np
from functools import reduce
from nltk.tokenize import word_tokenize
from collections import defaultdict
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tqdm
from torch.utils.data import DataLoader

class PreprocessTools:
    def __init__(self, vocab_path=None):
        self.word_count = defaultdict(int)       
        if vocab_path:
            self.word_transformer = load_file_json(vocab_path)
            self.vocab_size = len(self.word_transformer) + 1
        else:
            self.word_transformer = defaultdict(int)
            self.vocab_size = -1
        
    def tokenize(self, data):
        return [[word for word in word_tokenize(txt)] for txt in data]

    def make_word_transformar(self, train_data:list):
        for data in train_data:
            for word in data:
                self.word_count[word] += 1
        sorted_word_count = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, count) in enumerate(sorted_word_count):
            if count < 2:
                break
            else:
                self.word_transformer[word] = idx + 1
        self.vocab_size = len(self.word_transformer) + 1

    def txt2ids(self, txt_list:list):
        txt_ids = list()
        for txt in txt_list:
            ids = list()
            for word in txt:
                ids.append(self.word_transformer[word])
            txt_ids.append(ids)
        return txt_ids


    def ids2vec(self, txt_ids:list):
        txt_vec = list()
        identity = np.identity(self.vocab_size)
        for ids in txt_ids:
            txt_vec.append(identity[ids])
        return txt_vec
    
    
def load_data(path):
    with open(path, mode='r') as f:
        X = list()
        Y = list()
        for line in f:
            line = line.strip()
            splited_line = line.split('\t')
            X.append(splited_line[0])
            Y.append(splited_line[1])
        return X, Y

def save_file_json(path, data):
    with open(path, mode='w') as out_file:
        out_file.write(json.dumps(data)+'\n')
        
def load_file_json(path):
    with open(path, mode='r') as in_file:
        data = json.load(in_file)
    return data

def chr2num(y):
    converter = {'b':0, 't':1, 'e':2, 'm':3}
    return [converter[article_type] for article_type in y]

class MyRNN(torch.nn.Module):
    def __init__(self, vocab_size, dw=300, dh=50, L=4, num_layers=1, bidirectional=False, rnn_bias=True, PATH=None):
        super(MyRNN, self).__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dw, self.dh = dw, dh
        if PATH:
            self.embed = nn.from_pretrained(PATH)
        else:
            m = nn.Embedding(vocab_size, dw, padding_idx=0)
            nn.init.normal_(m.weight, mean=0, std=dw ** -0.5)
            nn.init.constant_(m.weight[0], 0)
            self.embed = m
        self.rnn = nn.RNN(dw, dh, bias=rnn_bias, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, nonlinearity='tanh')
        if bidirectional:
            self.linear = nn.Linear(2 * dh, L, bias=True)
        else:
            self.linear = nn.Linear(dh, L, bias=True)
        self.softmax = nn.LogSoftmax(dim=1) # dim=-1 or 1
        
    '''
    x: ids (not one hot vector)
    '''
    def forward(self, x):
        x = self.embed(x)
        _, hidden = self.rnn(x)
        hidden = hidden.view(self.num_layers, 2 if self.bidirectional else 1, -1, self.dh)
        last_hidden = hidden[-1]
        if self.bidirectional:
            x = self.linear(torch.cat([last_hidden[0], last_hidden[1]], dim=1))
        else:
            x = self.linear(last_hidden[0])
        x = self.softmax(x)
        return x 

class MyDataSets(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = [torch.LongTensor(data) for data in x]
        self.y = [torch.LongTensor([data]) for data in y]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
def collate_fn(batch):
    x = [data[0] for data in batch]
    x = nn.utils.rnn.pad_sequence(x, batch_first=True)
    y = torch.LongTensor([data[1] for data in batch])
    return x, y

    
def execution(data_x, data_y, op, criterion, model, batch_size=1, is_train=True, use_gpu=False):
    if is_train: model.train()
    else: model.eval()
    ndata = len(data_x)
    dataset = MyDataSets(data_x, data_y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    sum_loss, acc_score = 0, 0
    for batch_x, batch_y in data_loader:
        op.zero_grad()
        if use_gpu:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        if is_train:
            loss.backward()
            op.step()
        sum_loss += loss.data.item() * len(batch_x)
        pred = torch.argmax(out, dim=1)
        acc_score += np.sum((pred == batch_y).cpu().detach().numpy())
    return sum_loss / ndata, acc_score / ndata * 100


if __name__ == "__main__":
    preprocess = PreprocessTools('work/vocab.json')
    
    x_train = load_file_json('work/train_x.json')['data']
    y_train = np.asarray(load_file_json('work/train_y.json')['data'])
    x_valid = load_file_json('work/valid_x.json')['data']
    y_valid = np.asarray(load_file_json('work/valid_y.json')['data'])
    x_test = load_file_json('work/test_x.json')['data']
    y_test = np.asarray(load_file_json('work/test_y.json')['data'])


    vocab_size = preprocess.vocab_size
    torch.manual_seed(1234)
    model = MyRNN(vocab_size, dw=300, dh=50, L=4, num_layers=2, bidirectional=True)
    ntrain = len(x_train)
    nepoch = 10 
    batch_size = 128 
    op = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss() 

    train_writer = SummaryWriter(log_dir='./work/logs/train')
    valid_writer = SummaryWriter(log_dir='./work/logs/valid')
    logger = list()
    for epoch in tqdm(range(nepoch)):
        train_loss, train_acc = execution(x_train, y_train, op, criterion, model, batch_size=batch_size)
        train_writer.add_scalar("loss", train_loss, epoch) 
        train_writer.add_scalar("accuracy", train_acc, epoch)
        with torch.no_grad():
            valid_loss, valid_acc = execution(x_valid, y_valid, op, criterion, model, batch_size=batch_size, is_train=False)
            valid_writer.add_scalar("loss", valid_loss, epoch)
            valid_writer.add_scalar("accuracy", valid_acc, epoch)
        logger.append({'epoch':epoch, 'train_loss':train_loss, 'train_acc':train_acc, 'valid_loss':valid_loss, 'valid_acc':valid_acc})
        print({'epoch':epoch, 'train_loss':train_loss, 'train_acc':train_acc, 'valid_loss':valid_loss, 'valid_acc':valid_acc})
    train_writer.close()
    valid_writer.close()
