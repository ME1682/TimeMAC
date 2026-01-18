import os
import pandas as pd
import pathlib
import numpy as np
import argparse
import json
import boto3
# from hts import HTSRegressor
import joblib
import ast
import plotly.graph_objects as go
# from hts.hierarchy import HierarchyTree

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

#s3 = boto3.client("s3")
# Aggregate data by region and Quantity
def get_region_columns(df, region):
    return [col for col in df.columns if region in col]

def prepare_data(df_raw):
    print('******************* Prepare Data **********************')
    product = df_raw[(df_raw['item'] == "mens_clothing")]
    print(product.head())
    product["region_state"] = product.apply(lambda x: f"{x['region']}_{x['state']}", axis=1)
    region_states = product["region_state"].unique()
    grouped_sections = product.groupby(["region", "region_state"])
    edges_hierarchy = list(grouped_sections.groups.keys())
    # Now, we must not forget that total is our root node.
    second_level_nodes = product.region.unique()
    root_node = "total"
    root_edges = [(root_node, second_level_node) for second_level_node in second_level_nodes]
    root_edges += edges_hierarchy
    product_bottom_level = product.pivot(index="date", columns="region_state", values="quantity")
    regions = product["region"].unique().tolist()
    for region in regions:
        region_cols = get_region_columns(product_bottom_level, region)
        product_bottom_level[region] = product_bottom_level[region_cols].sum(axis=1)

    product_bottom_level["total"] = product_bottom_level[regions].sum(axis=1)
   
    # create hierarchy
    # Now that we have our dataset ready, let's define our hierarchy tree. 
    # We need a dictionary, where each key is a column (node) in our hierarchy and a list of its children.
    hierarchy = dict()

    for edge in root_edges:
        parent, children = edge[0], edge[1]
        hierarchy.get(parent)
        if not hierarchy.get(parent):
            hierarchy[parent] = [children]
        else:
            hierarchy[parent] += [children]
    
    product_bottom_level.index = pd.to_datetime(product_bottom_level.index)
    product_bottom_level = product_bottom_level.resample("D").sum()
    print('******************* End Prepare Data **********************')
    return hierarchy, product_bottom_level, region_states

# have a try
df_raw = pd.read_csv("retail-usa-clothing.csv",
                          parse_dates=True,
                          header=0,
                          names=['date', 'state',
                                   'item', 'quantity', 'region',
                                   'country']
                    )
df_raw = df_raw[(df_raw['item'] == "womens_clothing")]
print(df_raw.head())
state = ['NewYork','Alabama','NewJersey','Pennsylvania','Kentucky','Mississippi','Tennessee','Alaska','California',
          'Hawaii','Oregon','Illinois','Indiana','Ohio','Connecticut','Maine','RhodeIsland','Vermont'
          ]
region = ['Mid-Alantic','SouthCentral','Pacific','EastNorthCentral','NewEngland']
df_raw = df_raw.rename(columns={'date': 'ds', 'quantity': 'y'})
'''
df_train = df_raw.query(f'ds <= "2009-04-29"').copy()
df_train.to_csv("train.csv")
df_test = df_raw.query(f'ds > "2009-04-29"').copy()
df_test.to_csv("test.csv")
df_other = df_raw.copy()
df_other.to_csv("other.csv")
'''
# For Hier_e2e
df = df_raw
df = df.set_index(["ds", "state"])
df = df.groupby(level="state")
for name, group in df:
    group.to_csv(f'group/state_{name}.csv')

# create other hierarchy for hier_e2e
hierarchy_node = [['NewYork', 'Alabama', 'NewJersey', 'Pennsylvania', 'Kentucky', 'Mississippi', 'Tennessee',
                   'Alaska', 'California', 'Hawaii', 'Oregon', 'Illinois', 'Indiana', 'Ohio', 'Connecticut',
                   'Maine', 'RhodeIsland', 'Vermont'],
                  ['Mid-Alantic', 'SouthCentral', 'Pacific', 'EastNorthCentral', 'NewEngland'],
                  ['total']]
total_node_number = 24
hierarchy_0_node_number = 18
# 分层的层级总数
hierarchy_level = 3
hierarchy_name = ["state", "region"]
hierarchy = 1
dataset_name = 'amazon'
df_read = pd.read_csv(f"all.csv")
df_read = df_read.drop('item', axis=1)
df_read = df_read.drop('country', axis=1)
df_read = df_read.drop('number', axis=1)
df_read["ds"] = pd.to_datetime(df_read["ds"])
df_read = df_read.rename(columns={'region': 'unique_id'})
for cnt in hierarchy_node[hierarchy]:
    print("Hierarchy middle: " + cnt)
    df = df_read.loc[df_read['unique_id'] == cnt] # 抽出该层需要的节点的数据
    # 对该节点下层节点的数据求和并排序
    df = df.set_index(["ds", hierarchy_name[0]])
    df = df.groupby(level="ds").sum()
    df.to_csv(f'group/region_{cnt}.csv')
hierarchy = 2
for cnt in hierarchy_node[hierarchy]:
    print("Hierarchy 2, " + cnt)
    df = df_read.set_index(["ds", hierarchy_name[0]])
    df = df.groupby(level="ds").sum()
    df = df.assign(unique_id="total")
    df = df.sort_values(by='ds')
    df.to_csv(f'group/total.csv')