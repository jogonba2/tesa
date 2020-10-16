# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import BaseWrapperDataset
import torch
import json


class SoftLabelDataset(BaseWrapperDataset):

    def __init__(self, dataset, label_dictionary, soft_labels):
        super().__init__(dataset)
        self.offset = -label_dictionary.nspecial
        self.soft_labels = json.loads(soft_labels)
        self.reverse_label_dictionary = {value + self.offset: key
                                         for key, value in label_dictionary.indices.items()
                                         if value + self.offset >= 0}

    def __getitem__(self, idx):
        label = (self.dataset[idx] + self.offset).item()
        return torch.tensor([self.soft_labels[self.reverse_label_dictionary[label]]], dtype=torch.float32)