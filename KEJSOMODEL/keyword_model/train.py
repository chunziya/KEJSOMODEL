import os
import time
import argparse
import numpy as np
from os.path import join
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

import sys
print(sys.path)
sys.path.append(r"/Users/czc/PythonFile/KEJSOMODEL")

from keyword_model.data_loader import TextDataset, DataLoader
from keyword_model.model_1 import myLSTM

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='rnn', help="models used")
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=37, help='Random seed.')
parser.add_argument('--lr', type=float, default=5e-2, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--max-sequence-length', type=int, default=15, help="Max sequence length for raw sequences")

parser.add_argument('--batch', type=int, default=512, help="Batch size")
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--embedding-size', type=int, default=256, help="Embeding size for LSTM layer")
parser.add_argument('--hidden-size', type=int, default=256, help="Hidden size for LSTM layer")

parser.add_argument('--instance-normalization', action='store_true', default=True, help="Enable instance normalization")
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--train-ratio', type=float, default=85, help="Training ratio (0, 100)")
parser.add_argument('--test-ratio', type=float, default=15, help="Test ratio (0, 100)")
parser.add_argument('--class-weight-balanced', action='store_true', default=False, help="Adjust weights inversely proportional to class frequencies in the input data")

args = parser.parse_args()


# def cal_auc(y_true, prob_pred):
#     return roc_auc_score(y_true, prob_pred)


# def cal_f1(y_true, prob_pred):
#     f1 = 0
#     threshold = 0
#     # precision = 0
#     # recall = 0
#     pred = prob_pred.copy()
#     for thr in [j * 0.01 for j in range(100)]:
#         for i in range(len(prob_pred)):
#             pred[i] = prob_pred[i] > thr
#         performance = precision_recall_fscore_support(y_true, pred, average='binary')
#         if performance[2] > f1:
#             precision = performance[0]
#             recall = performance[1]
#             f1 = performance[2]
#             threshold = thr
#     return threshold, precision, recall, f1


# class Metrics(Callback):
#     def on_train_begin(self, logs={}):
#         self.val_f1s = []
#         # self.val_recalls = []
#         # self.val_precisions = []

#     def on_epoch_end(self, epoch, logs={}):
#         val_targ = self.validation_data[1]
#         val_predict = self.model.predict(self.validation_data[0])
#         threshold, precision, recall, f1 = cal_f1(val_targ, val_predict)
#         logger.info(
#             "test_set:\nbest_thr:{:2f}\t precision:{:4f}\trecall:{:4f}\tf1score:{:4f}\n".format(threshold, precision,recall, f1))
#         return


def train(train_loader, test_loader, model, args=args):
    model_chechpoint = ModelCheckpoint('out/model_1_1.h5', save_best_only=True, save_weights_only=False)

    history = model.fit(x=train_loader.data['keywords'],
                        y=train_loader.data['labels'],
                        batch_size=args.batch,
                        epochs=args.epochs,
                        validation_data=(test_loader.data['keywords'], 
                        test_loader.data['labels']),
                        callbacks=[model_chechpoint])
    logger.info(str(history))


def main(args=args):
    np.random.seed(args.seed)
    dataset = TextDataset(args.seed, args.shuffle, args.max_sequence_length, args.batch)

    train_loader, test_loader = dataset.split_dataset(args.test_ratio / 100)

    model = myLSTM(vocab_size=dataset.vocab_size,
                   max_sequence_length=args.max_sequence_length,
                   embedding_size=args.embedding_size,
                   hidden_size=args.hidden_size,
                   dropout=args.dropout)

    # Train model
    os.makedirs('out', exist_ok=True)
    t_total = time.time()
    logger.info("training...")
    train(train_loader, test_loader, model, args=args)
    logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))


if __name__ == '__main__':
    main(args=args)
