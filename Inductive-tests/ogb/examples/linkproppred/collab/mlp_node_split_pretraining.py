import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from logger import Logger
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(predictor, x, split_edge, optimizer, batch_size):
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        edge = pos_train_edge[perm].t()

        pos_out = predictor(x[edge[0]], x[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long,
                             device=x.device)
        neg_out = predictor(x[edge[0]], x[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(predictor, x, split_edge, evaluator, batch_size):
    predictor.eval()

    pos_train_edge = split_edge['train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
    
    y_pred_pos_list = list(np.array(pos_test_pred))
    y_pred_neg_list = list(np.array(neg_test_pred).flatten())
    print('AUROC: ', roc_auc_score([1] * len(y_pred_pos_list) + [0] * len(y_pred_neg_list), y_pred_pos_list + y_pred_neg_list))
    print('AUPRC: ', average_precision_score([1] * len(y_pred_pos_list) + [0] * len(y_pred_neg_list), y_pred_pos_list + y_pred_neg_list))

    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (MLP)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true') ## Not using the Node2vec embeddings by default as they create shortcuts
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100) ## Default was 200
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1) ## Defaulkt was 10
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-collab')
    split_edge = dataset.get_edge_split()
    data = dataset[0]

    ########################### Changing the datasets for random node split inductive test inspired by GraIL ######################

    pos_train_edge = split_edge['train']['edge'].to(device).cpu()
    pos_val_edge = split_edge['valid']['edge'].to(device).cpu()
    neg_val_edge = split_edge['valid']['edge_neg'].to(device).cpu()
    pos_test_edge = split_edge['test']['edge'].to(device).cpu()
    neg_test_edge = split_edge['test']['edge_neg'].to(device).cpu()

    all_edges_df = pd.concat([pd.DataFrame(pos_train_edge.numpy()), pd.DataFrame(pos_val_edge.numpy()),pd.DataFrame(pos_test_edge.numpy())])
    all_nodes = list(set(all_edges_df[0].tolist()).union(set(all_edges_df[1].tolist())))

    pos_train_nodes, pos_val_test_nodes = train_test_split(all_nodes, test_size=0.2, random_state=71139) #53517  #83959 #73183 #42876
    pos_val_nodes, pos_test_nodes  = train_test_split(pos_val_test_nodes, test_size=0.5, random_state=55054) #39830 #63909 #60742 #88434

    train_edges = all_edges_df[all_edges_df[0].isin(pos_train_nodes) & all_edges_df[1].isin(pos_train_nodes)]
    valid_edges = all_edges_df[all_edges_df[0].isin(pos_val_nodes) & all_edges_df[1].isin(pos_val_nodes)]
    test_edges = all_edges_df[all_edges_df[0].isin(pos_test_nodes) & all_edges_df[1].isin(pos_test_nodes)]

    split_edge['train']['edge'] = torch.tensor(train_edges.values)
    split_edge['valid']['edge'] = torch.tensor(valid_edges.values)
    split_edge['test']['edge'] = torch.tensor(test_edges.values)

    print('Edges in train, validation, and test: ', split_edge['train']['edge'].shape, split_edge['valid']['edge'].shape, split_edge['test']['edge'].shape)
    print('Edges lost: ', len(all_edges_df) - len(train_edges) - len(valid_edges) - len(test_edges))
    print('Nodes in train, validation, and test: ', len(pos_train_nodes), len(pos_val_nodes), len(pos_test_nodes))

    ################################################################################################################################

    x = data.x ## OGB provided Word2vec embeddings are pre-trained
    #print(args.use_node_embedding)
    #if args.use_node_embedding:
    #    embedding = torch.load('embedding.pt', map_location='cpu')
    #    x = torch.cat([x, embedding], dim=-1)
    x = x.to(device)

    predictor = LinkPredictor(x.size(-1), args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    for run in range(args.runs):
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(predictor, x, split_edge, optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(predictor, x, split_edge, evaluator,
                               args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()
