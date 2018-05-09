import argparse

import os
import sys
import json
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


PAD_TOKEN = '_PAD_'
UNK_TOKEN = '_UNK_'
random.seed(1024)
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
mydir = os.path.dirname(os.path.abspath(__file__))
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
# Methods for loading SST data

def read_dictionary_txt_with_phrase_ids(dictionary_path, phrase_ids_path):
    print('Reading data dictionary_path={} phrase_ids_path={}'.format(
    dictionary_path, phrase_ids_path))

    with open(phrase_ids_path) as f:
        phrase_ids = set(line.strip() for line in f)
    #print(phrase_ids)
  
    f=pd.read_csv(dictionary_path)
    examples_dict = dict()
    examples_dict_2=dict()
    for line in f.values:
      parts = line
      phrase = str(parts[3])
      phrase_2 = str(parts[4])
      
      phrase_id = str(parts[0])
      if phrase_id not in phrase_ids:
        continue

      example = dict()
      example['phrase'] = phrase.replace('(', '-LRB').replace(')', '-RRB-')
      example['tokens'] = example['phrase'].split(' ')
      example['example_id'] = phrase_id
      example['label'] = int(parts[5])
      examples_dict[example['example_id']] = example
        
      example_2 = dict()
      example_2['phrase'] = phrase_2.replace('(', '-LRB').replace(')', '-RRB-')
      example_2['tokens'] = example_2['phrase'].split(' ')
      example_2['example_id'] = phrase_id
      example_2['label'] = int(parts[5])
      #print(example_2)
      examples_dict_2[example_2['example_id']] = example_2


    examples = [ex for _, ex in examples_dict.items()]
    examples_2 = [ex for _, ex in examples_dict_2.items()]
    print('Found {} examples.'.format(len(examples)))
    return examples,examples_2


def build_vocab(datasets):
    vocab = dict()
    vocab[PAD_TOKEN] = len(vocab)
    vocab[UNK_TOKEN] = len(vocab)
    for data in datasets:
        for example in data:
          for word in example['tokens']:
            if word not in vocab:
              vocab[word] = len(vocab)

    print('Vocab size: {}'.format(len(vocab)))

    return vocab


class TokenConverter(object):
    def __init__(self, vocab):
        self.vocab = vocab
        self.unknown = 0

    def convert(self, token):
        if token in self.vocab:
          id = self.vocab.get(token)
        else:
          id = self.vocab.get(UNK_TOKEN)
          self.unknown += 1
        return id


def convert2ids(data, vocab):
    converter = TokenConverter(vocab)
    for example in data:
        example['tokens'] = list(map(converter.convert, example['tokens']))
    print('Found {} unknown tokens.'.format(converter.unknown))
    return data


def load_data_and_embeddings(data_path, phrase_ids_path, embeddings_path,embeddings_path2):
    dictionary_path = os.path.join(data_path)
    train_data_1,train_data_2 = read_dictionary_txt_with_phrase_ids(dictionary_path, os.path.join('train_ids.csv'))
    validation_data_1,validation_data_2 = read_dictionary_txt_with_phrase_ids(dictionary_path, os.path.join('validation_ids.csv'))
    test_data_1,test_data_2 = read_dictionary_txt_with_phrase_ids(dictionary_path, os.path.join('test_ids.csv'))
    vocab = build_vocab([train_data_1,train_data_2, validation_data_1,validation_data_2, test_data_1,test_data_2])    
    vocab, embeddings = load_embeddings(embeddings_path, vocab, cache=True)
    embeddings2=None
    if embeddings_path2:
        vocab, embeddings2,words = load_embeddings2_channel(embeddings_path2, vocab, cache=True)
        vocab1, embeddings = load_embeddings2_channel2(embeddings_path, vocab, words,cache=True)
    train_data_1 = convert2ids(train_data_1, vocab)
    train_data_2 = convert2ids(train_data_2, vocab)
    validation_data_2 = convert2ids(validation_data_2, vocab)
    validation_data_1 = convert2ids(validation_data_1, vocab)
    test_data_1 = convert2ids(test_data_1, vocab)
    test_data_2 = convert2ids(test_data_2, vocab)
    return train_data_1,train_data_2, validation_data_1,validation_data_2, test_data_1,test_data_2, vocab, embeddings,embeddings2


def load_embeddings(path, vocab, cache=False, cache_path=None):
    rows = []
    new_vocab = [UNK_TOKEN]

    if cache_path is None:
        cache_path = path + '.cache'

    # Use cache file if it exists.
    if os.path.exists(cache_path):
        path = cache_path

    print("Reading embeddings from {}".format(path))

  # first pass over the embeddings to vocab and relevant rows
    with open(path,'r', encoding='utf8') as f:
        for line in f:
          word, row = line.split(' ', 1)
          if word == UNK_TOKEN:
            raise ValueError('The unk token should not exist w.in embeddings.')
          if word in vocab:
            rows.append(line)
            new_vocab.append(word)

  # optionally save relevant rows to cache file.
    if cache and not os.path.exists(cache_path):
        with open(cache_path, 'w') as f:
          for line in rows:
            f.write(line)
          print("Cached embeddings to {}".format(cache_path))

    # turn vocab list into a dictionary
    new_vocab = {w: i for i, w in enumerate(new_vocab)}

    print('New vocab size: {}'.format(len(new_vocab)))

    assert len(rows) == len(new_vocab) - 1

    # create embeddings matrix
    embeddings = np.zeros((len(new_vocab), 300), dtype=np.float32)
    for i, line in enumerate(rows):
        embeddings[i+1] = list(map(float, line.strip().split(' ')[1:]))

    return new_vocab, embeddings


#####start from here
def load_embeddings2_channel(path, vocab, cache=False, cache_path=None):
    rows = []
    words=[]
    new_vocab = [UNK_TOKEN]

    if cache_path is None:
        cache_path = path + '.cache'

    # Use cache file if it exists.
    if os.path.exists(cache_path):
        path = cache_path

    print("Reading embeddings from {}".format(path))

  # first pass over the embeddings to vocab and relevant rows
    with open(path) as f:
        for line in f:
          word, row = line.split(' ', 1)
          if word == UNK_TOKEN:
            raise ValueError('The unk token should not exist w.in embeddings.')
          if word in vocab:
            rows.append(line)
            words.append(word)
            new_vocab.append(word)

  # optionally save relevant rows to cache file.
    if cache and not os.path.exists(cache_path):
        with open(cache_path, 'w') as f:
          for line in rows:
            f.write(line)
          print("Cached embeddings to {}".format(cache_path))

  # turn vocab list into a dictionary
    new_vocab = {w: i for i, w in enumerate(new_vocab)}

    print('New vocab size: {}'.format(len(new_vocab)))

    assert len(rows) == len(new_vocab) - 1

    # create embeddings matrix
    embeddings = np.zeros((len(new_vocab), 300), dtype=np.float32)
    for i, line in enumerate(rows):
        embeddings[i+1] = list(map(float, line.strip().split(' ')[1:]))

    return new_vocab, embeddings,words

def load_embeddings2_channel2(path, vocab,words, cache=False, cache_path=None):
    rows = []
    new_words=[]
    new_vocab = [UNK_TOKEN]

    if cache_path is None:
        cache_path = path + '.cache'

    # Use cache file if it exists.
    if os.path.exists(cache_path):
        path = cache_path

    print("Reading embeddings from {}".format(path))

  # first pass over the embeddings to vocab and relevant rows
    with open(path) as f:
        for line in f:
          word, row = line.split(' ', 1)
          if word == UNK_TOKEN:
            raise ValueError('The unk token should not exist w.in embeddings.')
          if word in vocab:
            rows.append(line)
            new_words.append(word)
            new_vocab.append(word)

  # optionally save relevant rows to cache file.
    if cache and not os.path.exists(cache_path):
        with open(cache_path, 'w') as f:
          for line in rows:
            f.write(line)
          print("Cached embeddings to {}".format(cache_path))

  # turn vocab list into a dictionary
    new_vocab = {w: i for i, w in enumerate(new_vocab)}

    print('New vocab size: {}'.format(len(new_vocab)))

    assert len(rows) == len(new_vocab) - 1

    # create embeddings matrix
    embeddings = np.zeros((len(new_vocab), 300), dtype=np.float32)
    for i, line in enumerate(rows):
        new_i=words.index(new_words[i])
        embeddings[new_i+1] = list(map(float, line.strip().split(' ')[1:]))

    return new_vocab, embeddings
# Batch Iterator

def prepare_data(data):
    # pad data
    maxlen = max(max(map(len, data)),5)
    data = [ex + [0] * (maxlen-len(ex)) for ex in data]

    # wrap in tensor
    return torch.LongTensor(data)


def prepare_labels(labels):
    try:
        return torch.LongTensor(labels)
    except:
        return labels


def batch_iterator(dataset,dataset_2, batch_size, forever=False):
    dataset_size = len(dataset)
    order = None
    nbatches = dataset_size // batch_size

    def init_order():
        return random.sample(range(dataset_size), dataset_size)

    def get_batch(start, end):
        batch = [dataset[ii] for ii in order[start:end]]
        data = prepare_data([ex['tokens'] for ex in batch])
        batch_2 = [dataset_2[ii] for ii in order[start:end]]
        data_2 = prepare_data([ex['tokens'] for ex in batch_2])
        labels = prepare_labels([ex['label'] for ex in batch])
        example_ids = [ex['example_id'] for ex in batch]
        return data, data_2,labels, example_ids

    order = init_order()

    while True:
        for i in range(nbatches):
          start = i*batch_size
          end = (i+1)*batch_size
          yield get_batch(start, end)

        if nbatches*batch_size < dataset_size:
          yield get_batch(nbatches*batch_size, dataset_size)

        if not forever:
          break
    
    order = init_order()


# Models

class BagOfWordsModel(nn.Module):
    def __init__(self, embeddings):
        super(BagOfWordsModel, self).__init__()
        self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1], sparse=True)
        self.embed.weight.data.copy_(torch.from_numpy(embeddings))
        self.classify = nn.Linear(embeddings.shape[1], 2)

    def forward(self, x):
        return self.classify(self.embed(x).sum(1))

class  CNNClassifier(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim=100, kernel_sizes=(3, 4, 5), dropout=0.5,multichannel=False,vocab_size2=None):
        super(CNNClassifier,self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        channel_number=2
        self.embedding2 = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])
        self.convs2 = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])

        # kernal_size = (K,D) 
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)
    
    
    def init_weights(self, pretrained_word_vectors, is_static=False, pretrained_word_vectors2=None):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_word_vectors))
        if is_static:
            self.embedding.weight.requires_grad = False
        self.embedding2.weight.data.copy_(torch.from_numpy(pretrained_word_vectors))
        if is_static:
            self.embedding2.weight.requires_grad = False


    def forward(self, inputs_1,inputs_2, is_training=False):
        inputs1 = self.embedding(inputs_1).unsqueeze(1) # (B,1,T,D)
        #print(inputs1)
        inputs2 = self.embedding2(inputs_2).unsqueeze(1)
        #print(inputs2)
        
        inputs1 = [F.relu(conv(inputs1)).squeeze(3) for conv in self.convs] #[(N,Co,W), ...]*len(Ks)
        inputs1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs1] #[(N,Co), ...]*len(Ks)
        inputs2 = [F.relu(conv(inputs2)).squeeze(3) for conv in self.convs2] #[(N,Co,W), ...]*len(Ks)
        inputs2 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs2] #[(N,Co), ...]*len(Ks)
        #print(len(inputs1[0]))
        #print(len(inputs2[0]))
        #diff=np.array(inputs1)-np.array(inputs2)
        concated = torch.cat(inputs1, 1)-torch.cat(inputs2, 1)
        #print(concated.shape)
        #print(concated.shape)
        #print(type(concated))
        #print(type(torch.cat(inputs1, 1)))
        if is_training:
            concated = self.dropout(concated) # (N,len(Ks)*Co)
        out = self.fc(concated) 
        return F.log_softmax(out,1)
# Utility Methods

def checkpoint_model(step, val_loss, model, opt, save_path):
    save_dict = dict(
        step=step,
        val_loss=val_loss,
        model_state_dict=model.state_dict(),
        opt_state_dict=opt.state_dict())
    torch.save(save_dict, save_path)


def load_model(model, opt, load_path):
    load_dict = torch.load(load_path)
    step = load_dict['step']
    val_loss = load_dict['val_loss']
    model.load_state_dict(load_dict['model_state_dict'])
    opt.load_state_dict(load_dict['opt_state_dict'])
    return step, val_loss


# Main
def run_validation(model, dataset,dataset_2, options):
    err = 0
    count = 0
    #model.train(False)
    for data, data_2,labels, _ in batch_iterator(dataset, dataset_2,options.batch_size, forever=False):
        if USE_CUDA:
            outp =model (Variable(data).cuda(),Variable(data_2).cuda())
            outp=outp.cpu()
        else:
            outp = model(Variable(data),Variable(data_2))
        loss = nn.CrossEntropyLoss()(F.log_softmax(outp,1), Variable(labels))
        acc = (outp.data.max(1)[1] == labels).sum() / data.shape[0]
        err += (1-acc) * data.shape[0]
        count += data.shape[0]
    err = err / count
    print('Loss={}, Ev-Err={}'.format(loss.data[0],err))
    return err


def run_test(model, dataset,dataset_2, options):
    print('Writing predictions to {}'.format(os.path.abspath(options.predictions)))
    preds_dict = dict()
    #model.train(False)
    for data,data_2, _, example_ids in batch_iterator(dataset,dataset_2, options.batch_size, forever=False):
        if USE_CUDA:
            outp =model (Variable(data).cuda(),Variable(data_2).cuda())
            outp=outp.cpu()
        else:
            outp = model(Variable(data),Variable(data_2))
        preds = outp.data.max(1)[1]

        for id, pred in zip(example_ids, preds):
          preds_dict[id] = pred

    with open(options.predictions, 'w') as f:
        for id, pred in preds_dict.items():
          f.write('{}|{}\n'.format(id, pred))


def run(options):
    if options.multi_channel:  
        train_data_1,train_data_2, validation_data_1,validation_data_2, test_data_1,test_data_2, vocab, embeddings,embeddings2 =     load_data_and_embeddings(options.data, options.ids, options.embeddings,options.embeddings2)
        size2=embeddings2.shape[0]
    else:
        train_data_1,train_data_2, validation_data_1,validation_data_2, test_data_1,test_data_2, vocab, embeddings,embeddings2 =     load_data_and_embeddings(options.data, options.ids, options.embeddings,None)
        size2=None
    if options.get_validation_result:
        print ('wrting actual validation result')
        with open('validation.txt', 'w') as f:
            for example in validation_data_1:
              f.write('{}|{}\n'.format(example['example_id'], example['label']))
    model = CNNClassifier(embeddings.shape[0],embeddings.shape[1],2,multichannel=options.multi_channel,vocab_size2=size2)
    model.init_weights(embeddings,options.static,embeddings2)
    if USE_CUDA:
        model = model.cuda()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    #model.train(True)
    step = 0
    best_val_err = 1
  
    if options.eval_only_mode:
        step, best_val_err = load_model(model, opt, options.model)
        print('Model loaded from {}\nstep={} best_val_err={}'.format(options.model, step, best_val_err))
        if options.get_validation_result:
            run_test(model, validation_data_1,validation_data_2, options)
        else:
            run_test(model, test_data_1,test_data_2, options)
        sys.exit()
  
    for data,data_2 ,labels, _ in batch_iterator(train_data_1, train_data_2,options.batch_size, forever=True):
        #print(1)
        if USE_CUDA:
            outp = model(Variable(data).cuda(),Variable(data_2).cuda(),True)
            label_input=Variable(labels).cuda()
            loss =  nn.CrossEntropyLoss()(F.log_softmax(outp,1), label_input).cuda()
            acc = (outp.cpu().data.max(1)[1] == labels).sum() / data.shape[0]
        else:
            outp = model(Variable(data),Variable(data_2),True)
            loss =  nn.CrossEntropyLoss()(F.log_softmax(outp,1), Variable(labels))
            acc = (outp.data.max(1)[1] == labels).sum() / data.shape[0]
        model.zero_grad()
        loss.backward()
        opt.step()

        if step % options.log_every == 0:
            print('Step={} Tr-Loss={} Tr-Acc={}'.format(step, loss.data[0], acc))

        if step % options.eval_every == 0:
            val_err = run_validation(model, validation_data_1,validation_data_2, options)

          # early stopping
        if val_err < best_val_err:
            best_val_err = val_err
            print('Checkpointing model step={} best_val_err={}.'.format(step, best_val_err))
            checkpoint_model(step, val_err, model, opt, options.model)

        step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', default=mydir, type=str)
    parser.add_argument('--data', default=os.path.expanduser('train.csv'), type=str)
    parser.add_argument('--embeddings', default=os.path.expanduser('glove.840B.300d.txt'), type=str)
    parser.add_argument('--model', default=os.path.join(mydir, 'model.ckpt'), type=str)
    parser.add_argument('--predictions', default=os.path.join(mydir, 'predictions.txt'), type=str)
    parser.add_argument('--log_every', default=100, type=int)
    parser.add_argument('--eval_every', default=1000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_only_mode', action='store_true')
    parser.add_argument('--multi_channel', action='store_true')
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--embeddings2', default=os.path.expanduser('GoogleNews-vectors-negative300.txt'), type=str)
    parser.add_argument('--get_validation_result', action='store_true')
    options = parser.parse_args()

    print(json.dumps(options.__dict__, sort_keys=True, indent=4))
  
    run(options)

