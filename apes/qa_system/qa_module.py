import numpy as np
import lasagne
import theano
import theano.tensor as T
import pickle
from os import environ
import sys, os
import time
from apes.qa_system import utils
import apes.qa_system.config
import logging
from apes.qa_system import nn_layers

base_file_path = os.path.dirname(os.path.abspath(__file__))


def gen_examples(x1, x2, l, y, batch_size):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = utils.get_minibatches(len(x1), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_x1 = [x1[t] for t in minibatch]
        mb_x2 = [x2[t] for t in minibatch]
        mb_l = l[minibatch]
        mb_y = [y[t] for t in minibatch]
        mb_x1, mb_mask1 = utils.prepare_data(mb_x1)
        mb_x2, mb_mask2 = utils.prepare_data(mb_x2)
        all_ex.append((mb_x1, mb_mask1, mb_x2, mb_mask2, mb_l, mb_y))
    return all_ex


def build_fn(args, embeddings):
    """
        Build training and testing functions.
    """
    in_x1 = T.imatrix('x1')
    in_x2 = T.imatrix('x2')
    in_mask1 = T.matrix('mask1')
    in_mask2 = T.matrix('mask2')
    in_l = T.matrix('l')
    in_y = T.ivector('y')

    l_in1 = lasagne.layers.InputLayer((None, None), in_x1)
    l_mask1 = lasagne.layers.InputLayer((None, None), in_mask1)
    l_emb1 = lasagne.layers.EmbeddingLayer(l_in1, args.vocab_size,
                                           args.embedding_size, W=embeddings)

    l_in2 = lasagne.layers.InputLayer((None, None), in_x2)
    l_mask2 = lasagne.layers.InputLayer((None, None), in_mask2)
    l_emb2 = lasagne.layers.EmbeddingLayer(l_in2, args.vocab_size,
                                           args.embedding_size, W=l_emb1.W)

    network1 = nn_layers.stack_rnn(l_emb1, l_mask1, args.num_layers, args.hidden_size,
                                   grad_clipping=args.grad_clipping,
                                   dropout_rate=args.dropout_rate,
                                   only_return_final=(args.att_func == 'last'),
                                   bidir=args.bidir,
                                   name='d',
                                   rnn_layer=args.rnn_layer)

    network2 = nn_layers.stack_rnn(l_emb2, l_mask2, args.num_layers, args.hidden_size,
                                   grad_clipping=args.grad_clipping,
                                   dropout_rate=args.dropout_rate,
                                   only_return_final=True,
                                   bidir=args.bidir,
                                   name='q',
                                   rnn_layer=args.rnn_layer)

    args.rnn_output_size = args.hidden_size * 2 if args.bidir else args.hidden_size

    if args.att_func == 'mlp':
        att = nn_layers.MLPAttentionLayer([network1, network2], args.rnn_output_size,
                                          mask_input=l_mask1)
    elif args.att_func == 'bilinear':
        att = nn_layers.BilinearAttentionLayer([network1, network2], args.rnn_output_size,
                                               mask_input=l_mask1)
    elif args.att_func == 'avg':
        att = nn_layers.AveragePoolingLayer(network1, mask_input=l_mask1)
    elif args.att_func == 'last':
        att = network1
    elif args.att_func == 'dot':
        att = nn_layers.DotProductAttentionLayer([network1, network2], mask_input=l_mask1)
    else:
        raise NotImplementedError('att_func = %s' % args.att_func)

    network = lasagne.layers.DenseLayer(att, args.num_labels,
                                        nonlinearity=lasagne.nonlinearities.softmax)

    if args.pre_trained is not None:
        checkpoint = utils.load_params(args.pre_trained)
        lasagne.layers.set_all_param_values(network, checkpoint[b'params'], trainable=True)
        del checkpoint[b'params']
        logging.debug('Loaded pre-trained model: %s' % args.pre_trained)
        for checkpoint_param in checkpoint.items():
            logging.debug(checkpoint_param)

    logging.debug('#params: %d' % lasagne.layers.count_params(network, trainable=True))
    for layer in lasagne.layers.get_all_layers(network):
        logging.debug(layer)

    # Test functions
    test_prob = lasagne.layers.get_output(network, deterministic=True) * in_l
    test_prediction = T.argmax(test_prob, axis=-1)
    acc = T.sum(T.eq(test_prediction, in_y))
    test_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_l, in_y], acc)

    # Train functions
    train_prediction = lasagne.layers.get_output(network) * in_l
    train_prediction = train_prediction / \
        train_prediction.sum(axis=1).reshape((train_prediction.shape[0], 1))
    train_prediction = T.clip(train_prediction, 1e-7, 1.0 - 1e-7)
    loss = lasagne.objectives.categorical_crossentropy(train_prediction, in_y).mean()
    # TODO: lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    params = lasagne.layers.get_all_params(network, trainable=True)

    if args.optimizer == 'sgd':
        updates = lasagne.updates.sgd(loss, params, args.learning_rate)
    elif args.optimizer == 'adam':
        updates = lasagne.updates.adam(loss, params)
    elif args.optimizer == 'rmsprop':
        updates = lasagne.updates.rmsprop(loss, params)
    else:
        raise NotImplementedError('optimizer = %s' % args.optimizer)
    train_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_l, in_y],
                               loss, updates=updates)

    return train_fn, test_fn, params


def eval_acc(test_fn, all_examples):
    """
        Evaluate accuracy on `all_examples`.
    """
    acc = 0
    n_examples = 0
    for x1, mask1, x2, mask2, l, y in all_examples:
        acc += test_fn(x1, mask1, x2, mask2, l, y)
        n_examples += len(x1)
    return acc * 100.0 / n_examples

def main(args):
    logging.debug('-' * 50)
    logging.debug('Load data files..')

    if args.debug:
        logging.debug('*' * 10 + ' Train')
        documents, questions, answers = utils.load_data(args.train_file, 100, relabeling=args.relabeling)
        logging.debug('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, 100, relabeling=args.relabeling)
    else:
        logging.debug('*' * 10 + ' Train')
        documents, questions, answers = utils.load_data(args.train_file, relabeling=args.relabeling)
        logging.debug('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, args.max_dev, relabeling=args.relabeling)

    args.num_train = len(documents)
    args.num_dev = len(dev_examples[0])

    logging.debug('-' * 50)
    logging.debug('Build dictionary..')
    word_dict = utils.build_dict(documents + questions)
    entity_markers = list(set([w for w in word_dict.keys()
                              if w.startswith('@entity')] + answers))
    entity_markers = ['<unk_entity>'] + entity_markers
    entity_dict = {w: index for (index, w) in enumerate(entity_markers)}

    # save entity dictionary
    # print('Saving entity dictionary, entity count {}'.format(len(entity_dict)))
    with open(os.path.join(base_file_path, 'entity_dict.pkl'), 'wb') as entity_f:
        pickle.dump(entity_dict, entity_f)

    with open(os.path.join(base_file_path, 'word_dict.pkl'), 'wb') as words_f:
        pickle.dump(word_dict, words_f)

    logging.debug('Entity markers: %d' % len(entity_dict))
    if args.pre_trained:
        args.num_labels = 328
    else:
        print(len(entity_dict))
        args.num_labels = len(entity_dict)

    logging.debug('-' * 50)
    # Load embedding file
    print(args)
    embeddings = utils.gen_embeddings(word_dict, args.embedding_size, args.embedding_file)
    (args.vocab_size, args.embedding_size) = embeddings.shape
    logging.debug('Compile functions..')
    train_fn, test_fn, params = build_fn(args, embeddings)
    logging.debug('Done.')
    if args.prepare_model:
        return args, word_dict, entity_dict, train_fn, test_fn, params

    logging.debug('-' * 50)
    logging.debug(args)

    logging.debug('-' * 50)
    logging.debug('Intial test..')
    dev_x1, dev_x2, dev_l, dev_y = utils.vectorize(dev_examples, word_dict, entity_dict)
    assert len(dev_x1) == args.num_dev
    all_dev = gen_examples(dev_x1, dev_x2, dev_l, dev_y, args.batch_size)
    dev_acc = eval_acc(test_fn, all_dev)
    logging.debug('Dev accuracy: %.2f %%' % dev_acc)
    best_acc = dev_acc

    if args.test_only:
        return

    utils.save_params(args.model_file, params, epoch=0, n_updates=0)

    # Training
    logging.debug('-' * 50)
    logging.debug('Start training..')
    train_x1, train_x2, train_l, train_y = utils.vectorize((documents, questions, answers), word_dict, entity_dict)
    assert len(train_x1) == args.num_train
    start_time = time.time()
    n_updates = 0

    all_train = gen_examples(train_x1, train_x2, train_l, train_y, args.batch_size)
    for epoch in range(args.num_epoches):
        np.random.shuffle(all_train)
        for idx, (mb_x1, mb_mask1, mb_x2, mb_mask2, mb_l, mb_y) in enumerate(all_train):
            logging.debug('#Examples = %d, max_len = %d' % (len(mb_x1), mb_x1.shape[1]))
            train_loss = train_fn(mb_x1, mb_mask1, mb_x2, mb_mask2, mb_l, mb_y)
            logging.debug('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)' %
                         (epoch, idx, len(all_train), train_loss, time.time() - start_time))
            n_updates += 1

            if n_updates % args.eval_iter == 0:
                samples = sorted(np.random.choice(args.num_train, min(args.num_train, args.num_dev),
                                                  replace=False))
                sample_train = gen_examples([train_x1[k] for k in samples],
                                            [train_x2[k] for k in samples],
                                            train_l[samples],
                                            [train_y[k] for k in samples],
                                            args.batch_size)
                logging.debug('Train accuracy: %.2f %%' % eval_acc(test_fn, sample_train))
                dev_acc = eval_acc(test_fn, all_dev)
                logging.debug('Dev accuracy: %.2f %%' % dev_acc)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    logging.debug('Best dev accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                 % (epoch, n_updates, dev_acc))
                    utils.save_params(args.model_file, params, epoch=epoch, n_updates=n_updates)


from collections import namedtuple
def qa_model(debug=False, test_only=False, prepare_model=False, 
    random_seed=1013, train_file=None, dev_file=None, pre_trained=None, model_file='model.pkl.gz', 
    log_file=None, embedding_file=None, max_dev=None, relabeling=True, 
    embedding_size=None, hidden_size=128, bidir=True, num_layers=1, rnn_type='gru', 
    att_func='bilinear', batch_size=32, num_epoches=100, eval_iter=100, dropout_rate=0.2, 
    optimizer='sgd', learning_rate=0.1, grad_clipping=10.0):

    args = namedtuple("args", "debug, test_only, prepare_model, random_seed, train_file, dev_file, pre_trained, model_file, log_file, embedding_file, max_dev, relabeling, embedding_size, hidden_size, bidir, num_layers, rnn_type, att_func, batch_size, num_epoches, eval_iter, dropout_rate, optimizer, learning_rate, grad_clipping")
    args.debug = debug
    args.test_only = test_only
    args.prepare_model = prepare_model
    args.random_seed = random_seed
    args.train_file = train_file
    args.dev_file = dev_file
    args.pre_trained = pre_trained
    args.model_file = model_file
    args.log_file = log_file
    args.embedding_file = embedding_file
    args.max_dev = max_dev
    args.relabeling = relabeling
    args.embedding_size = embedding_size
    args.hidden_size = hidden_size
    args.bidir = bidir
    args.num_layers = num_layers
    args.rnn_type = rnn_type
    args.att_func = att_func
    args.batch_size = batch_size
    args.num_epoches = num_epoches
    args.eval_iter = eval_iter
    args.dropout_rate = dropout_rate
    args.optimizer = optimizer
    args.learning_rate = learning_rate
    args.grad_clipping = grad_clipping
    
    # args = config.get_args()
    np.random.seed(args.random_seed)
    lasagne.random.set_rng(np.random.RandomState(args.random_seed))

    if args.train_file is None:
        raise ValueError('train_file is not specified.')

    if args.dev_file is None:
        raise ValueError('dev_file is not specified.')

    if args.rnn_type == 'lstm':
        args.rnn_layer = lasagne.layers.LSTMLayer
    elif args.rnn_type == 'gru':
        args.rnn_layer = lasagne.layers.GRULayer
    else:
        raise NotImplementedError('rnn_type = %s' % args.rnn_type)

    if args.embedding_file is not None:
        dim = utils.get_dim(args.embedding_file)
        if (args.embedding_size is not None) and (args.embedding_size != dim):
            raise ValueError('embedding_size = %d, but %s has %d dims.' %
                             (args.embedding_size, args.embedding_file, dim))
        args.embedding_size = dim
    elif args.embedding_size is None:
        raise RuntimeError('Either embedding_file or embedding_size needs to be specified.')

    if args.log_file is None:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=args.log_file,
                            filemode='w', level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    logging.debug(' '.join(sys.argv))
    return main(args)

def checkpoint_args(debug=False, test_only=False, prepare_model=False, 
    random_seed=1013, train_file=None, dev_file=None, pre_trained=None, model_file='model.pkl.gz', 
    log_file=None, embedding_file=None, max_dev=None, relabeling=True, 
    embedding_size=None, hidden_size=128, bidir=True, num_layers=1, rnn_type='gru', 
    att_func='bilinear', batch_size=32, num_epoches=100, eval_iter=100, dropout_rate=0.2, 
    optimizer='sgd', learning_rate=0.1, grad_clipping=10.0):

    args = namedtuple("args", "debug, test_only, prepare_model, random_seed, train_file, dev_file, pre_trained, model_file, log_file, embedding_file, max_dev, relabeling, embedding_size, hidden_size, bidir, num_layers, rnn_type, att_func, batch_size, num_epoches, eval_iter, dropout_rate, optimizer, learning_rate, grad_clipping")
    args.debug = debug
    args.test_only = test_only
    args.prepare_model = prepare_model
    args.random_seed = random_seed
    args.train_file = train_file
    args.dev_file = dev_file
    args.pre_trained = pre_trained
    args.model_file = model_file
    args.log_file = log_file
    args.embedding_file = embedding_file
    args.max_dev = max_dev
    args.relabeling = relabeling
    args.embedding_size = embedding_size
    args.hidden_size = hidden_size
    args.bidir = bidir
    args.num_layers = num_layers
    args.rnn_type = rnn_type
    args.att_func = att_func
    args.batch_size = batch_size
    args.num_epoches = num_epoches
    args.eval_iter = eval_iter
    args.dropout_rate = dropout_rate
    args.optimizer = optimizer
    args.learning_rate = learning_rate
    args.grad_clipping = grad_clipping
    
    # args = config.get_args()
    np.random.seed(args.random_seed)
    lasagne.random.set_rng(np.random.RandomState(args.random_seed))

    if args.train_file is None:
        raise ValueError('train_file is not specified.')

    if args.dev_file is None:
        raise ValueError('dev_file is not specified.')

    if args.rnn_type == 'lstm':
        args.rnn_layer = lasagne.layers.LSTMLayer
    elif args.rnn_type == 'gru':
        args.rnn_layer = lasagne.layers.GRULayer
    else:
        raise NotImplementedError('rnn_type = %s' % args.rnn_type)

    if args.embedding_file is not None:
        dim = utils.get_dim(args.embedding_file)
        if (args.embedding_size is not None) and (args.embedding_size != dim):
            raise ValueError('embedding_size = %d, but %s has %d dims.' %
                             (args.embedding_size, args.embedding_file, dim))
        args.embedding_size = dim
    elif args.embedding_size is None:
        raise RuntimeError('Either embedding_file or embedding_size needs to be specified.')

    if args.log_file is None:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=args.log_file,
                            filemode='w', level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    logging.debug(' '.join(sys.argv))
    return args

def test(args, word_dict, entity_dict, train_fn, test_fn, params):
    dev_examples = utils.load_data(args.dev_file, args.max_dev, relabeling=args.relabeling)
    dev_x1, dev_x2, dev_l, dev_y = utils.vectorize(dev_examples, word_dict, entity_dict)
    assert len(dev_x1) == args.num_dev
    all_dev = gen_examples(dev_x1, dev_x2, dev_l, dev_y, args.batch_size)
    dev_acc = eval_acc(test_fn, all_dev)
    return dev_acc


def load_model(embedding_file, model_file='model.pkl.gz',  entity_dictionry_filename='entity_dict.pkl', words_dictionry_filename='word_dict.pkl'):

    args = checkpoint_args(embedding_file=embedding_file,model_file=model_file, train_file='None2', dev_file='None' )

    with open(os.path.join(base_file_path, entity_dictionry_filename), 'rb') as entity_f:
        entity_dict = pickle.load(entity_f)
        # print('{} entities found!'.format(len(entity_dict)))

    with open(os.path.join(base_file_path, words_dictionry_filename), 'rb') as entity_f:
        word_dict = pickle.load(entity_f)

    logging.debug('Entity markers: %d' % len(entity_dict))
    if args.pre_trained:
        args.num_labels = 328
    else:
        args.num_labels = len(entity_dict)
    # Load embedding file
    embeddings = utils.gen_embeddings(word_dict, args.embedding_size, args.embedding_file)
    (args.vocab_size, args.embedding_size) = embeddings.shape
    logging.debug('Compile functions..')
    train_fn, test_fn, params = build_fn(args, embeddings)
    logging.debug('Done.')
    return args, word_dict, entity_dict, train_fn, test_fn, params