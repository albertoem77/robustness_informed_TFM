"""
@author: Carlos Loucera
"""

from itertools import chain, repeat
from pathlib import Path

import numpy as np
import pandas as pd

from isrobust_TFG.utils import get_resource_path


def get_reactome_adj(pth=None):
    """
    Parse a gmt file to a decoupler pathway dataframe.
    """
    # Adapted from Single-cell best practices book

    if pth is None:
        pth = get_resource_path("c2.cp.reactome.v7.5.1.symbols.gmt")

    pathways = {}

    with Path(pth).open("r") as f:
        for line in f:
            name, _, *genes = line.strip().split("\t")
            pathways[name] = genes

    reactome = pd.DataFrame.from_records(
        chain.from_iterable(zip(repeat(k), v) for k, v in pathways.items()),
        columns=["geneset", "genesymbol"],
    )

    reactome = (
        reactome.drop_duplicates()
        .assign(belongs_to=1)
        .pivot(columns="geneset", index="genesymbol", values="belongs_to")
        .fillna(0)
    )

    return reactome


def read_circuit_names():
    path = get_resource_path("circuit_names.tsv.tar.xz")
    circuit_names = pd.read_csv(path, sep="\t")
    circuit_names.hipathia_id = circuit_names.hipathia_id.str.replace(" ", ".")
    circuit_names["effector"] = circuit_names.name.str.split(": ").str[-1]
    # circuit_names = circuit_names.set_index("hipathia_id")

    return circuit_names


def read_circuit_adj(with_effectors=False, gene_list=None):
    path = get_resource_path("pbk_circuit_hsa_sig.tar.xz")
    adj = pd.read_csv(path, sep=",", index_col=0)
    adj.index = adj.index.str.upper()
    if not with_effectors:
        adj = 1 * (adj > 0)

    if gene_list is not None:
        adj = adj.loc[adj.index.intersection(gene_list), :]

    adj.columns = adj.columns.str.replace(" ", ".")
    to_remove = adj.columns.str.contains("hsa04218")
    adj = adj.loc[:, ~to_remove]

    return adj


def build_pathway_adj_from_circuit_adj(circuit_adj):
    tmp_adj = circuit_adj.T
    tmp_adj.index.name = "circuit"
    tmp_adj = tmp_adj.reset_index()
    tmp_adj["pathway"] = tmp_adj.circuit.str.split("-").str[1]
    tmp_adj = tmp_adj.drop("circuit", axis=1)
    adj = 1 * tmp_adj.groupby("pathway").any()

    return adj


def build_pathway_adj_from_circuit_adj(circuit_adj):
    tmp_adj = circuit_adj.T
    tmp_adj.index.name = "circuit"
    tmp_adj = tmp_adj.reset_index()
    tmp_adj["pathway"] = tmp_adj.circuit.str.split("-").str[1]
    tmp_adj = tmp_adj.drop("circuit", axis=1)
    adj = 1 * tmp_adj.groupby("pathway").any()

    return adj


def build_circuit_pathway_adj(circuit_adj, pathway_adj):
    return (1 * (pathway_adj.dot(circuit_adj) > 0)).T


def get_adj_matrices(gene_list=None):
    circuit_adj = read_circuit_adj(with_effectors=False, gene_list=gene_list)
    pathway_adj = build_pathway_adj_from_circuit_adj(circuit_adj)
    circuit_to_pathway = build_circuit_pathway_adj(circuit_adj, pathway_adj)

    return circuit_adj, circuit_to_pathway


def get_random_adj(frac, shape, size, seed, index):
    np.random.RandomState(seed)
    random_layer = np.random.binomial(1, frac, size=size)
    random_layer = random_layer.reshape(shape)
    random_layer_names = [f"rand-{icol:02d}" for icol in range(random_layer.shape[1])]
    random_layer = pd.DataFrame(random_layer, index=index, columns=random_layer_names)
    random_layer = random_layer.loc[random_layer.any(axis=1), :]
    random_layer = random_layer.loc[:, random_layer.any(axis=0)]
    random_layer_names = random_layer.columns

    return random_layer, random_layer_names


def build_hipathia_renamers():
    circuit_names = read_circuit_names()
    circuit_names = circuit_names.rename(
        columns={"name": "circuit_name", "hipathia_id": "circuit_id"}
    )
    circuit_names["pathway_id"] = circuit_names["circuit_id"].str.split("-").str[1]
    circuit_names["pathway_name"] = circuit_names["circuit_name"].str.split(":").str[0]
    circuit_renamer = circuit_names.set_index("circuit_id")["circuit_name"].to_dict()
    pathway_renamer = circuit_names.set_index("pathway_id")["pathway_name"].to_dict()
    circuit_to_effector = (
        circuit_names.set_index("circuit_name")["effector"].str.strip().to_dict()
    )

    return circuit_renamer, pathway_renamer, circuit_to_effector


def sync_gexp_adj(gexp, adj):
    gene_list = adj.index.intersection(gexp.columns)
    gexp = gexp.loc[:, gene_list]
    adj = adj.loc[gene_list, :]

    return gexp, adj
