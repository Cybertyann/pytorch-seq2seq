import os
import argparse
import logging

import torch
import torch.optim.lr_scheduler
import torchtext

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity, NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#     # resuming from a specific checkpoint
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#       --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=11,
                    help="Random seed")
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--test_path', action='store', dest='test_path',
                    help='Path to test data')
parser.add_argument("--tag_scheme", default="iobes",
                    help="Tagging scheme (IOB or IOBES)")
parser.add_argument("--lower", action='store_false', default=False,
                    help="Lowercase words (this will not affect character inputs)")
parser.add_argument("--zeros", action='store_false', default=False,
                    help="Replace digits with 0")
parser.add_argument("--word_dim", type=int, default=300,
                    help="Token embedding dimension")
parser.add_argument("--word_lstm_dim", type=int, default=300,
                    help="Token LSTM hidden layer size")
parser.add_argument("--word_bidirect", action='store_true', default=True,
                    help="Use a bidirectional LSTM for words")
parser.add_argument("--pre_emb", action='store', default="glove.840B",
                    help="Location or name of pretrained embeddings")
parser.add_argument("--input_dropout", type=float, default=0,
                    help="Droupout on the input (0 = no dropout)")
parser.add_argument("--dropout", type=float, default=0.2,
                    help="Droupout on the output (0 = no dropout)")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory'
                         'has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_false', dest='resume', default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level', default='info',
                    help='Logging level.')
opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(
        os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    # SourceField requires that batch_first and include_lengths be true.
    src = SourceField(lower=opt.lower)
    # TargetField requires that batch_first be true as well as prepends <sos> and appends <eos> to sequences.
    tgt = TargetField()
    # Sequence's length cannot exceed max_len.
    max_len = 100
    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len
    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    test = torchtext.data.TabularDataset(
        path=opt.test_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    src.build_vocab(train, dev, test, max_size=50000)
    tgt.build_vocab(train, dev, test, max_size=1000)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    batch_iterator = torchtext.data.BucketIterator(
        dataset=train, batch_size=8,
        sort_key=lambda x: -len(x.src), repeat=False)
    for i, batch in enumerate(batch_iterator):
        input_variables, input_lengths = getattr(batch, 'src')
        target_variables = getattr(batch, 'tgt')
        print('Train inspection')
        for j, l in enumerate(input_lengths):
            src_indices = input_variables[j][0:l].data.tolist()
            tgt_indices = target_variables[j][0:l+2].data.tolist()
            print(' '.join([src.vocab.itos[k] for k in src_indices]))
            print(' '.join([tgt.vocab.itos[k] for k in tgt_indices]))
        if i == 1:
            break

    # inputs = torchtext.Field(lower=True, include_lengths=True, batch_first=True)
    # inputs.build_vocab(src.vocab)
    # src.vocab.load_vectors(wv_type='glove.6B', wv_dim=opt.word_dim)
    src.vocab.load_vectors(wv_type=opt.pre_emb, wv_dim=opt.word_dim)
    # src.vocab.load_vectors(wv_type='fasttext.en.300d', wv_dim=300)
    # src.vocab.load_vectors(wv_type='charngram.100d', wv_dim=100)
    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    # loss = Perplexity(weight, pad)
    loss = NLLLoss(weight=weight, mask=pad, size_average=True)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = opt.word_lstm_dim
        bidirectional = opt.word_bidirect
        encoder = EncoderRNN(vocab_size=len(src.vocab),
                             max_len=max_len,
                             word_dim=opt.word_dim,
                             hidden_size=hidden_size,
                             input_dropout_p=opt.input_dropout,
                             bidirectional=bidirectional,
                             n_layers=1,
                             rnn_cell='gru',
                             variable_lengths=True)
        decoder = DecoderRNN(vocab_size=len(tgt.vocab),
                             max_len=max_len,
                             hidden_size=hidden_size * 2 if bidirectional else 1,
                             dropout_p=opt.dropout,
                             use_attention=True,
                             bidirectional=bidirectional,
                             n_layers=1,
                             rnn_cell='gru',
                             eos_id=tgt.eos_id,
                             sos_id=tgt.sos_id)
        seq2seq = Seq2seq(encoder, decoder)
        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)
            print(param.data[0:3])
        _, _, norm_val = encoder.vectors_stats()
        encoder.init_vectors(src.vocab.vectors)
        # encoder.scale_vectors(0.08)
        encoder.normalize_vectors(norm_val)
        encoder.vectors_stats()
        for param in seq2seq.parameters():
            print(param.data[0:3])

        if torch.cuda.is_available():
            seq2seq.cuda()

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters(), lr=0.001), max_grad_norm=5)
        # optimizer = Optimizer(torch.optim.SGD(seq2seq.parameters(), lr=0.01, momentum=0.9), max_grad_norm=5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer.optimizer, step_size=10, gamma=0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer.optimizer, mode='min', factor=0.5, patience=5,
            verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=opt.batch_size, random_seed=opt.random_seed,
                          checkpoint_every=1000000, print_every=50, expt_dir=opt.expt_dir)
    seq2seq = t.train(seq2seq, train,
                      num_epochs=100, dev_data=dev, test_data=test,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)
    # teacher_forcing_ratio=0.5,

# predictor = Predictor(seq2seq, input_vocab, output_vocab)

# while True:
#     seq_str = raw_input("Type in a source sequence:")
#     seq = seq_str.strip().split()
#     print(predictor.predict(seq))