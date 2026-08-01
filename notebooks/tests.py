#!/usr/bin/env python
# coding: utf-8

# ## Path setting
# This cell should be run only one in one runtime

# In[ ]:

import os
# from sys import path

# path.append("..")

# from pathlib import Path

# os.chdir(Path.cwd().parent)


# # Imports and parameters

# In[ ]:


import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import psutil
from typing import List, Tuple

from src.graph_utils.dataset import Dataset
from src.graph_utils.reading import read_graph6
from src.descriptors.edge_descriptors import edge_descriptors_dict
from src.descriptors.node_descriptors import node_descriptors_dict
from src.collision_tests.comparison import TestOperator

from src.settings import Settings


# In[ ]:


SINGLE_FEATURES: bool     = True
CHOSEN_FEATURES: List[Tuple[str, ...]] = [('betweenness',), ('betweenness_normalized',)]

BENCHMARK_BREC: bool      = False


PLAIN_DESCRIPTORS: bool   = True
K_GRAPH2: bool            = True
K_GRAPH3: bool            = True
MODK_GRAPH3: bool         = True
COMBINED_K_GRAPH3: bool   = True


# In[43]:


for filename in os.listdir(Settings.raw_datasets_dir):
    if '.txt' in filename:
        old_filename = os.path.join('raw_datasets', filename)
        new_filename = re.sub(r'.txt', '', old_filename)
        print(old_filename, new_filename)

        os.rename(old_filename, new_filename)


# In[44]:


print(psutil.cpu_count())
print(len(os.sched_getaffinity(0)))
N_CORES = min(psutil.cpu_count(), len(os.sched_getaffinity(0)))


# ## Feature selection
# Generates lists of combinations of desciptors to test

# In[45]:


base_features = []
if CHOSEN_FEATURES:
    base_features = CHOSEN_FEATURES.copy()

elif SINGLE_FEATURES:
    single_features = [(k, ) for k in edge_descriptors_dict.keys()]

    node_features = [(f, ) for f in list(filter(lambda k: 'ldp' not in k, node_descriptors_dict.keys()))]
    node_features.extend([tuple(filter(lambda k: 'ldp' in k and 'normalized' not in k, node_descriptors_dict.keys()))])
    node_features.extend([tuple(filter(lambda k: 'ldp' in k and ('normalized' in k or 'degree' in k), node_descriptors_dict.keys()))])

    single_features.extend(node_features)
    base_features = single_features

else:
    raise ValueError("What else you want in features?")


all_features = []

if PLAIN_DESCRIPTORS:
    all_features.extend(base_features)
if K_GRAPH2:
    all_features.extend(list(map(lambda params: tuple(map(lambda x: x + ':k_graph2', params)), base_features)))
if K_GRAPH3:
    all_features.extend(list(map(lambda params: tuple(map(lambda x: x + ':k_graph3', params)), base_features)))
if MODK_GRAPH3:
    all_features.extend(list(map(lambda params: tuple(map(lambda x: x + ':modk_graph3', params)), base_features)))
if COMBINED_K_GRAPH3:
    all_features.extend(list(map(lambda features: tuple(x + k_graph for k_graph in ['',':k_graph2',':k_graph3'] for x in features), base_features)))

all_features


# ## Dataset Generation
# Here every dataset is loaded, and if k_graph cache does not exist, it's generated.

# In[ ]:


import os
from typing import List
from src.graph_utils.dataset import Dataset
from src.collision_tests.utils import TestParameters

arguments_list = []
datasets: List[Dataset] = []
# for dataset_name in ['graph5', 'graph6', 'graph7']:
# for dataset_name in ['regular']:
if BENCHMARK_BREC:
    dataset_names = ['brec_400']
else:
    dataset_names = list(map(lambda x: '.'.join(x.split('.')[:-1]), filter(lambda x: '.g6' in x and 'brec' not in x, os.listdir(Settings.raw_datasets_dir))))
for dataset_name in dataset_names:
    print(dataset_name)
    dataset = Dataset(name=dataset_name)

    datasets.append(dataset)
    dataset.get_metadata()


    for features in all_features:

        features = features
        arguments_list.append(TestParameters(dataset_name=dataset_name,
                                    features=features))


# In[ ]:


for d in datasets:
    print(len(d),'\t', d.name)


# In[ ]:


from tqdm import tqdm
for dataset in tqdm(datasets):
    dataset.generate_k_graph_cache(k=2)
    dataset.generate_k_graph_cache(k=3)
    dataset.generate_k_graph_cache(k=3, modified=True)


# In[ ]:


test_operator = TestOperator(single_thread=False, batch_size=N_CORES//2)

test_operator.tests(arguments_list)

