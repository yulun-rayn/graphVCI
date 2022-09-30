from typing import Callable, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset


class Dataset(InMemoryDataset):
    r"""Dataset

    Args:
        split (string): The type of dataset split
            (:obj:`"public"`, :obj:`"full"`, :obj:`"random"`).
            If set to :obj:`"public"`, the split will be the public fixed split
            from the `"Revisiting Semi-Supervised Learning with Graph
            Embeddings" <https://arxiv.org/abs/1603.08861>`_ paper.
            If set to :obj:`"full"`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"random"` split. (default: :obj:`20`)
        num_val (int, optional): The number of validation samples in case of
            :obj:`"random"` split. (default: :obj:`500`)
        num_test (int, optional): The number of test samples in case of
            :obj:`"random"` split. (default: :obj:`1000`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    Stats:
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1

            * - Name
              - #nodes
              - #edges
              - #features
              - #classes
            * - Cora
              - 2,708
              - 10,556
              - 1,433
              - 7
            * - CiteSeer
              - 3,327
              - 9,104
              - 3,703
              - 6
            * - PubMed
              - 19,717
              - 88,648
              - 500
              - 3
    """

    def __init__(self, data_path: Optional[str] = None, 
                 data: Optional[Data] = None, split: str = "random",
                 val_size: float = .1, test_size: float = .2,
                 num_train_per_class: int = 20, num_val: int = 500,
                 num_test: int = 1000, root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):

        assert data_path is not None or data is not None
        assert not (data_path is not None and data is not None)

        self.split = split.lower()
        if data is None:
            assert self.split in ['public', 'full', 'random']
        else:
            assert self.split in ['full', 'random']

        super().__init__(root, transform, pre_transform)
        if data_path is not None:
            self.data, self.slices = torch.load(data_path)
            data = self.get(0)
        else:
            num_nodes = data.x.size(0)

            if data.y is None:
                data.y = torch.ones(data.x.size(0), device = data.x.device)

            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            self.data, self.slices = data, None
            data = self.get(0)

        if split == 'full':
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            num_nodes = data.x.size(0)

            num_val = int(num_nodes * val_size)
            num_test = int(num_nodes * test_size)
            num_train = num_nodes - num_val - num_test
            num_train_per_class = int(num_train / self.num_classes)

            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])


class LabelData:
    def __init__(self, label_edges, labels):
        self.label_edges = label_edges
        self.labels = labels

    def __getitem__(self, i):
        return (self.label_edges[i], self.labels[i])

    def __len__(self):
        return len(self.labels)
