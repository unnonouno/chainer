import collections
import datetime
import time

import numpy as np

from chainer import cuda, Variable, FunctionSet
from chainer.cuda import to_gpu
import chainer.functions  as F
import chainer.optimizers as O
from chainer.functions import basic_math

use_gpu = True
n_unit = 400
window = 5
n_epoch = 10
minibatch_size = 100

if use_gpu:
    cuda.init()

index2word = {}
word2index = {}
counts = collections.defaultdict(lambda: 0)
dataset = []
with open('ptb.train.txt') as f:
    for line in f:
        for word in line.split():
            if word not in word2index:
                ind = len(word2index)
                word2index[word] = ind
                index2word[ind] = word
            counts[word2index[word]] += 1
            dataset.append(word2index[word])

n_vocab = len(word2index)

print('n_vocab: %d' % n_vocab)
print('data length: %d' % len(dataset))

model = FunctionSet(
    embed=F.EmbedID(n_vocab, n_unit),
    #l=F.Linear(n_unit, n_vocab),
    #l=F.HierarchicalSoftmax(n_unit, parent_node),
    #l=F.BinaryHierarchicalSoftmax(n_unit, tree),
    l=F.NegativeSampling(n_unit, [counts[w] for w in range(len(counts))], 20),
)

if use_gpu:
    model.to_gpu()

dataset = np.array(dataset, dtype=np.int32)
def continuous_bow(dataset, position):
    h = None
    es = []
    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        weight = 1.0 - float(np.abs(offset) - 1) / window
        d = dataset[position + offset]
        if use_gpu:
            d = to_gpu(d)
        x = Variable(d)
        e = model.embed(x) * weight
        es.append(e)

        if h is None:
            h = e
        else:
            h = h + e

    d = dataset[position]
    if use_gpu:
        d = to_gpu(d)
    t = Variable(d)
    #loss = softmax_cross_entropy(model.l(h), t)
    loss = model.l(h, t) / len(es)
    return loss


optimizer = O.Adam()
optimizer.setup(model.collect_parameters())

begin_time = time.time()
word_count = 0
skip = (len(dataset) - window * 2) / minibatch_size
next_count = 100000
for epoch in range(n_epoch):
    accum_loss = 0
    print 'epoch: %d' % epoch
    for i in range(0, skip):
        if word_count > next_count:
            now = time.time()
            duration = datetime.timedelta(seconds = now - begin_time)
            throuput = float(word_count) / (now - begin_time)
            print word_count, duration, throuput
            next_count += 100000

        position = np.array(range(0, minibatch_size)) * skip + (i + window)
        loss = continuous_bow(dataset, position)
        accum_loss += loss.data
        word_count += minibatch_size

        optimizer.zero_grads()
        loss.backward()
        optimizer.update()

    print accum_loss

model.to_cpu()
w = model.embed.W
for q in ['mr.', 'two', 'he', 'this']:
    v = w[word2index[q]]
    similarity = w.dot(v) / np.sqrt((w * w).sum(1)) / np.sqrt((v * v).sum())
    print 'query: %s' % q
    count = 0
    for i in (-similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        print index2word[i], similarity[i]
        count += 1
        if count == 5:
            break
    print


