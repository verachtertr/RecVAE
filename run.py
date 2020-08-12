import numpy as np

import torch
from torch import optim

import random
import scipy.sparse
from copy import deepcopy

from utils import load_data, split_into_train_and_test, ndcg, recall
from model import VAE
from recpack.metrics import NDCGK, RecallK

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--hidden-dim', type=int, default=600)
parser.add_argument('--latent-dim', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=500)
parser.add_argument('--beta', type=float, default=None)
parser.add_argument('--gamma', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--n-epochs', type=int, default=50)
parser.add_argument('--n-enc_epochs', type=int, default=3)
parser.add_argument('--n-dec_epochs', type=int, default=1)
parser.add_argument('--not-alternating', type=bool, default=False)
parser.add_argument('--user-col', type=str, default='user_id')
parser.add_argument('--item-col', type=str, default='item_id')
parser.add_argument('--timestamp-col', type=str, default='timestamp')
parser.add_argument('--min-users-per-item', type=int, default=3)
parser.add_argument('--min-items-per-user', type=int, default=3)
parser.add_argument('--t', type=float, default=None)


args = parser.parse_args()

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0")

# Load data using recpack

data = load_data(
    args.dataset,
    args.item_col,
    args.user_col,
    args.timestamp_col,
    args.min_users_per_item,
    args.min_items_per_user
)

t = args.t
if t is None:
    t = data.timestamps.min() + ((data.timestamps.max() - data.timestamps.min()) * 0.7)

train_data, valid_in_data, valid_out_data, test_in_data, test_out_data = split_into_train_and_test(
    data, t)
# subselect the nonzero users:
train_users = set(train_data.nonzero()[0])
validation_users = set(valid_in_data.nonzero()[0]).intersection(
    valid_out_data.nonzero()[0])
test_users = set(test_in_data.nonzero()[0]).intersection(
    test_out_data.nonzero()[0]
)

train_data = train_data[list(train_users)]
valid_in_data = valid_in_data[list(validation_users)]
valid_out_data = valid_out_data[list(validation_users)]

test_in_data = test_in_data[list(test_users)]
test_out_data = test_out_data[list(test_users)]

# Log the shapes and nnz of all the data segments
print(f"train_data - shape: {train_data.shape} -- nnz: {train_data.nnz}")
print(
    f"valid_in_data - shape: {valid_in_data.shape} -- nnz: {valid_in_data.nnz}")
print(
    f"valid_out_data - shape: {valid_out_data.shape} -- nnz: {valid_out_data.nnz}")
print(f"test_in_data - shape: {test_in_data.shape} -- nnz: {test_in_data.nnz}")
print(
    f"test_out_data - shape: {test_out_data.shape} -- nnz: {test_out_data.nnz}")


def generate(batch_size, device, data_in, data_out=None,
             shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1

    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)

    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)

    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield Batch(device, idx, data_in, data_out)


class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out

    def get_idx(self):
        return self._idx

    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)

    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]

    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)


def evaluate(model, data_in, data_out, metrics,
             samples_perc_per_epoch=1, batch_size=500):
    metrics = deepcopy(metrics)
    model.eval()

    for m in metrics:
        m['score'] = []

    for batch in generate(batch_size=batch_size,
                          device=device,
                          data_in=data_in,
                          data_out=data_out,
                          samples_perc_per_epoch=samples_perc_per_epoch
                          ):

        ratings_in = batch.get_ratings_to_dev()
        ratings_out = batch.get_ratings(is_out=True)

        ratings_pred = model(
            ratings_in,
            calculate_loss=False).cpu().detach().numpy()

        if not (data_in is data_out):
            ratings_pred[batch.get_ratings().nonzero()] = -np.inf

        for m in metrics:
            m['score'].append(m['metric'](ratings_pred, ratings_out, k=m['k']))

    for m in metrics:
        m['score'] = np.concatenate(m['score']).mean()

    return [x['score'] for x in metrics]


def evaluate_recpack(model, data_in, data_out, metrics,
                     samples_perc_per_epoch=1, batch_size=500):
    metrics = deepcopy(metrics)
    model.eval()
    print(f"shape of output: {data_out.shape}")
    full_expected = scipy.sparse.lil_matrix(data_out.shape)
    full_predicted = scipy.sparse.lil_matrix(data_out.shape)
    for i, batch in enumerate(generate(
        batch_size=batch_size,
        device=device,
        data_in=data_in,
        data_out=data_out,
        samples_perc_per_epoch=samples_perc_per_epoch
    )):

        ratings_in = batch.get_ratings_to_dev()
        ratings_out = batch.get_ratings(is_out=True)
        print(ratings_in.shape, ratings_out.shape)

        ratings_pred = model(
            ratings_in,
            calculate_loss=False).cpu().detach().numpy()

        start = i * batch_size
        end = (i * batch_size) + batch_size
        print(ratings_pred.shape)
        full_predicted[start:end] = ratings_pred
        full_expected[start:end] = ratings_out

    for m in metrics:
        m.calculate(full_expected.tocsr(), full_predicted.tocsr())

    return [x.value for x in metrics]


def run(model, opts, train_data, batch_size,
        n_epochs, beta, gamma, dropout_rate):
    model.train()
    for epoch in range(n_epochs):
        for batch in generate(batch_size=batch_size,
                              device=device, data_in=train_data, shuffle=True):
            ratings = batch.get_ratings_to_dev()

            for optimizer in opts:
                optimizer.zero_grad()

            _, loss = model(ratings, beta=beta, gamma=gamma,
                            dropout_rate=dropout_rate)
            loss.backward()

            for optimizer in opts:
                optimizer.step()


model_kwargs = {
    'hidden_dim': args.hidden_dim,
    'latent_dim': args.latent_dim,
    'input_dim': train_data.shape[1]
}
metrics = [{'metric': ndcg, 'k': 100}]

best_ndcg = -np.inf
train_scores, valid_scores = [], []

model = VAE(**model_kwargs).to(device)
model_best = VAE(**model_kwargs).to(device)

learning_kwargs = {
    'model': model,
    'train_data': train_data,
    'batch_size': args.batch_size,
    'beta': args.beta,
    'gamma': args.gamma
}

decoder_params = set(model.decoder.parameters())
encoder_params = set(model.encoder.parameters())

optimizer_encoder = optim.Adam(encoder_params, lr=args.lr)
optimizer_decoder = optim.Adam(decoder_params, lr=args.lr)


for epoch in range(args.n_epochs):

    if args.not_alternating:
        run(opts=[optimizer_encoder, optimizer_decoder],
            n_epochs=1, dropout_rate=0.5, **learning_kwargs)
    else:
        run(opts=[optimizer_encoder],
            n_epochs=args.n_enc_epochs,
            dropout_rate=0.5,
            **learning_kwargs)
        model.update_prior()
        run(opts=[optimizer_decoder],
            n_epochs=args.n_dec_epochs,
            dropout_rate=0,
            **learning_kwargs)

    train_scores.append(
        evaluate(model, train_data, train_data, metrics, 0.01)[0]
    )
    valid_scores.append(
        evaluate(model, valid_in_data, valid_out_data, metrics, 1)[0]
    )

    if valid_scores[-1] > best_ndcg:
        best_ndcg = valid_scores[-1]
        model_best.load_state_dict(deepcopy(model.state_dict()))

    print(f'epoch {epoch} | valid ndcg@100: {valid_scores[-1]:.4f} | ' +
          f'best valid: {best_ndcg:.4f} | train ndcg@100: {train_scores[-1]:.4f}')


test_metrics = [NDCGK(100), RecallK(20), RecallK(50)]

final_scores = evaluate_recpack(
    model_best,
    test_in_data,
    test_out_data,
    test_metrics)

for metric, score in zip(test_metrics, final_scores):
    print(f"{metric.name}:\t{score:.4f}")
