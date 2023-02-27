import os
import json
import time
import logging
from datetime import datetime
from collections import defaultdict

import torch

import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling

from gvci.dataset import Dataset, LabelData
from gvci.evaluate import evaluate_auc
from gvci.model.graph import MyLinkPred

from vci.utils.general_utils import initialize_logger, ljson

hparams = {
    "latent_dim": 128,
    "encoder_width": 128,
    "encoder_depth": 2,
    "lr": 3e-4
}

def prepare(args, hparams, state_dict=None):
    """
    Instantiates autoencoder and dataset to run an experiment.
    """
    device = (
        "cuda:" + str(args["gpu"])
            if (not args["cpu"]) 
                and torch.cuda.is_available() 
            else 
        "cpu"
    )

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                        add_negative_train_samples=False),
    ])

    graph = torch.load(args["graph_path"])
    # TODO: edge weight logits
    # TODO: dense graph

    datasets = Dataset(data=graph, transform=transform, split="random")

    model = MyLinkPred(
        sizes=[graph.x.size(1)] 
            + [hparams["encoder_width"]] * hparams["encoder_depth"],
        mode=args["graph_mode"],
        final_act="relu"
    ).to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model, datasets

def train_iter(data, model, optimizer, criterion, batch_size=128):
    model.train()

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        data.edge_label,
        data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    dataset = LabelData(edge_label_index.T, edge_label)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    losses = 0
    for label_batch in data_loader:
        label_index, label = label_batch[0].T, label_batch[1]

        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, label_index).view(-1)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()

    return losses / len(data_loader)

def link_pred(args, return_model=False):
    """
    Trains a link prediction model
    """
    if args["seed"] is not None:
        torch.manual_seed(args["seed"])

    if args["hparams"] != "":
        if isinstance(args["hparams"], str):
            with open(args["hparams"]) as f:
                dictionary = json.load(f)
            hparams.update(dictionary)
        else:
            hparams.update(args["hparams"])

    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    save_dir = os.path.join(args["artifact_path"], "saves/" + args["name"] + "_" + dt)
    os.makedirs(save_dir, exist_ok=True)

    initialize_logger(save_dir)
    ljson({"training_args": args})
    ljson({"model_params": hparams})
    logging.info("")

    # prepare
    model, datasets = prepare(args, hparams)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams["lr"])
    criterion = torch.nn.BCEWithLogitsLoss()

    train_data, val_data, test_data = datasets[0]

    # train
    start_time = time.time()
    best_val_auc = final_test_auc = 0
    for epoch in range(args["max_epochs"]):
        loss = train_iter(
            train_data, model, optimizer, criterion,
            batch_size=args["batch_size"]
        )
        val_auc = evaluate_auc(val_data, model)
        test_auc = evaluate_auc(test_data, model)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc

            torch.save(
                (model.state_dict(), args),
                os.path.join(
                    save_dir,
                    "best_val_model_seed={}.pt".format(args["seed"], epoch),
                ),
            )

        ellapsed_minutes = (time.time() - start_time) / 60

        # decay learning rate if necessary
        # also check stopping condition: patience ran out OR
        # time ran out OR max epochs reached
        stop = (
            (epoch == args["max_epochs"] - 1) 
            or (ellapsed_minutes > args["max_minutes"])
        )

        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            ljson(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "val_auc": val_auc,
                    "test_auc": test_auc,
                    "ellapsed_minutes": ellapsed_minutes,
                }
            )

            torch.save(
                (model.state_dict(), args),
                os.path.join(
                    save_dir,
                    "model_seed={}_epoch={}.pt".format(args["seed"], epoch),
                ),
            )

    ljson(f'Final Test: {final_test_auc:.4f}')

    z = model.encode(test_data.x, test_data.edge_index)
    final_edge_index = model.decode_all(z)

    if return_model:
        return final_edge_index, model
    else:
        return final_edge_index
