import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


expected_labels = []
sample_dataset = LungDataset(path, None)

for _, label in tqdm(sample_dataset):
    if np.any(label):
        expected_labels.append(1)
    else:
        expected_labels.append(0)

num_labels = np.unique(expected_labels, return_counts=True)
# print(num_labels)
fraction = num_labels[1][0] / num_labels[1][1]
weight_list = []

for label in expected_labels:
    if label == 0:
        weight_list.append(1)
    else:
        weight_list.append(fraction)

sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_list, len(weight_list))
sampler_loader = DataLoader(sample_dataset, sampler=sampler, batch_size=4)
# for data, label in sampler_loader:
#     print(label.sum([1,2,3]))
