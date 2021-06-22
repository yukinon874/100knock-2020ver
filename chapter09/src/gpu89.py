
import numpy as np
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from transformers import BertTokenizer, BertModel
import tqdm
import json


def load_data(path):
    with open(path, mode='r') as f:
        X = list()
        for line in f:
            line = line.strip()
            splited_line = line.split('\t')
            X.append(splited_line[0])
        return X

def load_file_json(path):
    with open(path, mode='r') as in_file:
        data = json.load(in_file)
    return data

class MyBertModel(torch.nn.Module):
    def __init__(self, L=4, dropout=0.2):
        super(MyBertModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.LogSoftmax(dim=1)
        self.linear = nn.Linear(768, L)
        
        
    def forward(self, inputs):
        out = self.bert(inputs['ids'], attention_mask=inputs['mask'])
        out = self.linear(self.dropout(out['pooler_output']))
        out = self.softmax(out)
        return out

                    
class MyDataSets(Dataset):
    def __init__(self, X, Y, tokenizer, max_len):
        self.X = X
        self.Y = Y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        text = self.X[idx]
        inputs = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_len,
          pad_to_max_length=True,
          truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
          'ids': torch.LongTensor(ids),
          'mask': torch.LongTensor(mask),
          'labels': torch.LongTensor(self.Y[idx])
        }

def execution(dataset, op, criterion, model, batch_size=1, is_train=True, use_gpu=False):
    if is_train: model.train()
    else: model.eval()
    ndata = len(dataset)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    sum_loss, acc_score = 0, 0
    for data in data_loader:
        op.zero_grad()
        labels = data['labels'].reshape(-1)
        out = model(data)
        loss = criterion(out, labels)
        if is_train:
            loss.backward()
            op.step()
        sum_loss += loss.data.item() * len(labels)
        pred = torch.argmax(out, dim=1)
        acc_score += np.sum((pred == labels).cpu().detach().numpy())
    return sum_loss / ndata, acc_score / ndata * 100

if __name__ == "__main__":
    x_train = load_data('data/train.txt')
    x_valid = load_data('data/valid.txt')
    x_test, = load_data('data/test.txt')
    y_train = np.asarray(load_file_json('work/train_y.json')['data']).reshape(-1, 1)
    y_valid = np.asarray(load_file_json('work/valid_y.json')['data']).reshape(-1, 1)
    y_test = np.asarray(load_file_json('work/test_y.json')['data']).reshape(-1, 1)

    max_len = 24
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset_train = MyDataSets(x_train, y_train, tokenizer, max_len)
    dataset_valid = MyDataSets(x_valid, y_valid, tokenizer, max_len)
    dataset_test = MyDataSets(x_test, y_test, tokenizer, max_len)
    
    torch.manual_seed(1234)
    model = MyBertModel(L=4, dropout=0.2)
    nepoch = 10 
    batch_size = 256 
    op = optim.Adagrad(model.parameters(), lr=0.00001)
    criterion = nn.NLLLoss() 


    logger = list()
    max_valid = -1
    max_model_param = None
    for epoch in tqdm.tqdm(range(nepoch)):
        train_loss, train_acc = execution(dataset_train, op, criterion, model, batch_size=batch_size)
        with torch.no_grad():
            valid_loss, valid_acc = execution(dataset_valid, op, criterion, model, batch_size=batch_size, is_train=False)

        if max_valid < valid_acc:
            max_valid = valid_acc
            max_model_param = model.state_dict()

        logger.append({'epoch':epoch, 'train_loss':train_loss, 'train_acc':train_acc, 'valid_loss':valid_loss, 'valid_acc':valid_acc})
        print({'epoch':epoch, 'train_loss':train_loss, 'train_acc':train_acc, 'valid_loss':valid_loss, 'valid_acc':valid_acc})
    
    model.load_state_dict(max_model_param)
    with torch.no_grad():
        test_loss, test_acc = execution(dataset_test, op, criterion, model, batch_size=batch_size, is_train=False)
        print(test_acc)
