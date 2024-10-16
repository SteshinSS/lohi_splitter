from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from .utils import get_similar_mols


def select_distinct_clusters(
    smiles, threshold, min_cluster_size, max_clusters, values, std_threshold
):
    """
    A greedy algorithm to select independent clusters from datasets. A part of the Lo splitter.
    """

    clusters = []
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    while len(clusters) < max_clusters:
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        all_fps = [fp_generator.GetFingerprint(x) for x in mols]
        total_neighbours = []
        stds = []

        for fps in all_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fps, all_fps)
            neighbors_idx = np.array(sims) > threshold
            total_neighbours.append(neighbors_idx.sum())
            stds.append(values[neighbors_idx].std())

        total_neighbours = np.array(total_neighbours)
        stds = np.array(stds)

        # Find the most distant cluster
        central_idx = None
        least_neighbours = max(total_neighbours)
        for idx, n_neighbours in enumerate(total_neighbours):
            if n_neighbours > min_cluster_size:
                if n_neighbours < least_neighbours:
                    if stds[idx] > std_threshold:
                        least_neighbours = n_neighbours
                        central_idx = idx

        if central_idx is None:
            break  # there are no clusters

        sims = DataStructs.BulkTanimotoSimilarity(all_fps[central_idx], all_fps)
        is_neighbour = np.array(sims) > threshold

        # Add them into cluster
        cluster_smiles = []
        for idx, value in enumerate(is_neighbour):
            if value:
                if (
                    idx != central_idx
                ):  # we add the central molecule at the end of the list
                    cluster_smiles.append(smiles[idx])
        cluster_smiles.append(smiles[central_idx])
        clusters.append(cluster_smiles)

        # Remove neighbours of neighbours from the rest of smiles
        nearest_sim = get_similar_mols(smiles, cluster_smiles)
        rest_idx = []
        for idx, dist in enumerate(nearest_sim):
            if dist < threshold:
                rest_idx.append(idx)
        smiles = smiles[rest_idx]
        values = values[rest_idx]

    return clusters, smiles


def lo_train_test_split(
    smiles, threshold, min_cluster_size, max_clusters, values, std_threshold
):
    """
    Lo splitter. Refer to tutorial 02_lo_split.ipynb and the paper by Simon Steshin titled "Lo-Hi: Practical ML Drug Discovery Benchmark", 2023.

    Parameters:
        smiles -- list of smiles
        threshold --  molecules with similarity larger than this number are considered similar
        min_cluster_size -- number of molecules per cluster
        max_clusters -- maximum number of selected clusters. The remaining molecules go to the training set.
        values -- values of the smiles
        std_threshold -- Lower bound of the acceptable standard deviation for a cluster. It should be greater than measurement noise.
                         If you're using ChEMBL-like data, set it to 0.60 for logKi and 0.70 for logIC50.
                         Set it lower if you have a high-quality dataset. Refer to the paper, Appendix B.

    Returns:
        clusters -- list of lists of smiles.
        train_smiles -- list of train smiles
    """
    if not isinstance(smiles, np.ndarray):
        smiles = np.array(smiles)
    if not isinstance(values, np.ndarray):
        values = np.array(values)

    cluster_smiles, train_smiles = select_distinct_clusters(
        smiles, threshold, min_cluster_size, max_clusters, values, std_threshold
    )
    train_smiles = list(train_smiles)
    # Move one molecule from each test cluster to the training set
    leave_one_clusters = []
    for cluster in cluster_smiles:
        train_smiles.append(cluster[-1])
        leave_one_clusters.append(cluster[:-1])

    return leave_one_clusters, train_smiles


def set_cluster_columns(data, cluster_smiles, train_smiles):
    data = data.copy()
    data["cluster"] = -1
    is_train = data["smiles"].isin(train_smiles)
    data.loc[is_train, ["cluster"]] = 0

    for i, cluster in enumerate(cluster_smiles):
        is_cluster = data["smiles"].isin(cluster)
        data.loc[is_cluster, ["cluster"]] = i + 1

    is_in_cluster = data["cluster"] != -1
    return data[is_in_cluster]
