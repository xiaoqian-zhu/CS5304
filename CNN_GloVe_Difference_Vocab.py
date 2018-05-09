
# coding: utf-8

# In[6]:


"""
sample code for assign4.py
load_sst can be used to read the files from sst, which can be downloaded from this link:
  https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
  
load_embeddings can be used to read files in the text format. Here's a link to
  word2vec - https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
  GloVe (300D 6B) - http://nlp.stanford.edu/data/glove.840B.300d.zip
The word2vec file is saved in a binary format and will need to be converted to text format. This can be done by installing gensim:
  pip install --upgrade gensim
  
Then running this snippet:
  from gensim.models.keyedvectors import KeyedVectors
  model = KeyedVectors.load_word2vec_format('path/to/GoogleNews-vectors-negative300.bin', binary=True)
  model.save_word2vec_format('path/to/GoogleNews-vectors-negative300.txt', binary=False)
To train:
  python assign4.py
To write test predictions:
  python assign4.py --eval_only_mode
"""

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
    for line in f.values:
      parts = line
      phrase = str(parts[2])
      phrase_id = str(parts[0])
      if phrase_id not in phrase_ids:
        continue

      example = dict()
      example['phrase'] = phrase.replace('(', '-LRB').replace(')', '-RRB-')
      example['tokens'] = example['phrase'].split(' ')
      example['example_id'] = phrase_id
      example['label'] = int(parts[1])
      examples_dict[example['example_id']] = example

    examples = [ex for _, ex in examples_dict.items()]

    print('Found {} examples.'.format(len(examples)))
    return examples


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
    train_data = read_dictionary_txt_with_phrase_ids(dictionary_path, os.path.join('train_ids.csv'))
    validation_data = read_dictionary_txt_with_phrase_ids(dictionary_path, os.path.join('validation_ids.csv'))
    test_data = read_dictionary_txt_with_phrase_ids(dictionary_path, os.path.join('test_ids.csv'))
    vocab = build_vocab([train_data, validation_data, test_data])    
    vocab, embeddings = load_embeddings(embeddings_path, vocab, cache=True)
    embeddings2=None
    if embeddings_path2:
        vocab, embeddings2,words = load_embeddings2_channel(embeddings_path2, vocab, cache=True)
        vocab1, embeddings = load_embeddings2_channel2(embeddings_path, vocab, words,cache=True)
    train_data = convert2ids(train_data, vocab)
    validation_data = convert2ids(validation_data, vocab)
    test_data = convert2ids(test_data, vocab)
    return train_data, validation_data, test_data, vocab, embeddings,embeddings2


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


def batch_iterator(dataset, batch_size, forever=False):
    dataset_size = len(dataset)
    order = None
    nbatches = dataset_size // batch_size

    def init_order():
        return random.sample(range(dataset_size), dataset_size)

    def get_batch(start, end):
        batch = [dataset[ii] for ii in order[start:end]]
        data = prepare_data([ex['tokens'] for ex in batch])
        labels = prepare_labels([ex['label'] for ex in batch])
        example_ids = [ex['example_id'] for ex in batch]
        return data, labels, example_ids

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
        channel_number=1
        self.MULTI=multichannel
        if self.MULTI:
            channel_number=2
            self.embedding2 = nn.Embedding(vocab_size2, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(channel_number, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])

        # kernal_size = (K,D) 
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)
    
    
    def init_weights(self, pretrained_word_vectors, is_static=False, pretrained_word_vectors2=None):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_word_vectors))
        if is_static:
            self.embedding.weight.requires_grad = False
        if self.MULTI:
            self.embedding2.weight.data.copy_(torch.from_numpy(pretrained_word_vectors2))
            if is_static:
                self.embedding2.weight.requires_grad = False


    def forward(self, inputs, is_training=False):
        inputs1 = self.embedding(inputs).unsqueeze(1) # (B,1,T,D)
        if self.MULTI:
            inputs2 = self.embedding2(inputs).unsqueeze(1)
            inputs1=torch.cat((inputs1, inputs2), 1)
        inputs1 = [F.relu(conv(inputs1)).squeeze(3) for conv in self.convs] #[(N,Co,W), ...]*len(Ks)
        inputs1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs1] #[(N,Co), ...]*len(Ks)
        concated = torch.cat(inputs1, 1)

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

def run_validation(model, dataset, options):
    err = 0
    count = 0
    #model.train(False)
    for data, labels, _ in batch_iterator(dataset, options.batch_size, forever=False):
        if USE_CUDA:
            outp =model (Variable(data).cuda())
            outp=outp.cpu()
        else:
            outp = model(Variable(data))
        #print(F.softmax(outp,1))
        #print('-----')
        loss = nn.CrossEntropyLoss()(outp, Variable(labels))
        #print(outp.data)
        acc = (outp.data.max(1)[1] == labels).sum() / data.shape[0]
        err += (1-acc) * data.shape[0]
        count += data.shape[0]
    err = err / count
    print('Loss={}, Ev-Err={}'.format(loss.data[0],err))
    return loss.data[0]


def run_test(model, dataset, options):
    print('Writing predictions to {}'.format(os.path.abspath(options.predictions)))
    preds_dict = dict()
    #model.train(False)
    for data, _, example_ids in batch_iterator(dataset, options.batch_size, forever=False):
        if USE_CUDA:
            outp =model (Variable(data).cuda())
            outp=outp.cpu()
        else:
            outp = model(Variable(data))
        preds = outp.data.max(1)[1]

        for id, pred in zip(example_ids, preds):
          preds_dict[id] = pred

    with open(options.predictions, 'w') as f:
        for id, pred in preds_dict.items():
          f.write('{}|{}\n'.format(id, pred))


def run(options):
    if options.multi_channel:  
        train_data, validation_data, test_data, vocab, embeddings ,embeddings2 =     load_data_and_embeddings(options.data, options.ids, options.embeddings,options.embeddings2)
        size2=embeddings2.shape[0]
    else:
        train_data, validation_data, test_data, vocab, embeddings ,embeddings2 =     load_data_and_embeddings(options.data, options.ids, options.embeddings,None)
        size2=None
    if options.get_validation_result:
        print ('wrting actual validation result')
        with open('validation.txt', 'w') as f:
            for example in validation_data:
              f.write('{}|{}\n'.format(example['example_id'], example['label']))
    model = CNNClassifier(embeddings.shape[0],embeddings.shape[1],2,multichannel=options.multi_channel,vocab_size2=size2)
    model.init_weights(embeddings,options.static,embeddings2)
    if USE_CUDA:
        model = model.cuda()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    #model.train(True)
    step = 0
    best_val_err = 1
    best_val_loss=10
    if options.eval_only_mode:
        step, best_val_loss = load_model(model, opt, options.model)
        print('Model loaded from {}\nstep={} best_val_loss={}'.format(options.model, step, best_val_loss))
        if options.get_validation_result:
            run_test(model, validation_data, options)
        else:
            run_test(model, test_data, options)
        sys.exit()
  
    for data, labels, _ in batch_iterator(train_data, options.batch_size, forever=True):
        if USE_CUDA:
            outp = model(Variable(data).cuda(),True)
            label_input=Variable(labels).cuda()
            loss =  nn.CrossEntropyLoss()(outp, label_input).cuda()
            acc = (outp.cpu().data.max(1)[1] == labels).sum() / data.shape[0]
        else:
            outp = model(Variable(data),True)
            loss =  nn.CrossEntropyLoss()(outp, Variable(labels))
            acc = (outp.data.max(1)[1] == labels).sum() / data.shape[0]
        model.zero_grad()
        loss.backward()
        opt.step()

        if step % options.log_every == 0:
            print('Step={} Tr-Loss={} Tr-Acc={}'.format(step, loss.data[0], acc))

        if step % options.eval_every == 0:
            val_loss = run_validation(model, validation_data, options)

          # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print('Checkpointing model step={} best_val_loss={}.'.format(step, best_val_loss))
            checkpoint_model(step, val_loss, model, opt, options.model)

        step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', default=mydir, type=str)
    parser.add_argument('--data', default=os.path.expanduser('processed.csv'), type=str)
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

