from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from .utils import get_similar_mols
from tqdm import tqdm
from scipy.sparse import csr_matrix


def construct_similarity_matrix(smiles, threshold, verbose=False):
    """
    Construct a similarity matrix for a given list of SMILES.
    """
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    # Convert SMILES to fingerprints
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    mols_iter = tqdm(mols, desc="Generating fingerprints") if verbose else mols

    all_fps = [fp_generator.GetFingerprint(mol) for mol in mols_iter]
    num_mols = len(all_fps)

    # Store only values above threshold to optimize memory usage
    rows, cols, data = [], [], []
    all_fps_iter = tqdm(all_fps, desc="Computing pairwise similarities") if verbose else all_fps

    for i, fp in enumerate(all_fps_iter):
        sims = DataStructs.BulkTanimotoSimilarity(fp, all_fps)
        valid_idx = np.where(np.array(sims) > threshold)[0]

        rows.extend([i] * len(valid_idx))
        cols.extend(valid_idx)
        data.extend(sims[j] for j in valid_idx)

    # CSR: Compressed Sparse Row format
    similarity_matrix = csr_matrix((data, (rows, cols)), shape=(num_mols, num_mols))
    return similarity_matrix



def select_distinct_clusters(
    smiles, threshold, min_cluster_size, max_clusters, values, std_threshold, verbose=False
):
    """
    A greedy algorithm to select independent clusters from datasets. A part of the Lo splitter.
    """
    clusters = []
    num_mols = len(smiles)
    similarity_matrix = construct_similarity_matrix(smiles, threshold, verbose=verbose)

    while len(clusters) < max_clusters:
        if verbose:
            print(f"Clusters: {len(clusters)}/{max_clusters}")

        # Compute neighbor counts and compute std deviations
        total_neighbours = np.array(similarity_matrix.getnnz(axis=1))
        stds = np.array([values[similarity_matrix[i].indices].std() if total_neighbours[i] > 0 else 0 
                         for i in range(num_mols)])

        # Select valid clusters indices for iteration
        valid_neighbours_idx = np.where(total_neighbours > min_cluster_size)[0]
        valid_stds_idx = np.where(stds > std_threshold)[0]
        valid_idx = np.intersect1d(valid_neighbours_idx, valid_stds_idx)

        # Find the most distant cluster considering cluster size and std deviation
        central_idx = None
        least_neighbours = max(total_neighbours)
        for idx in valid_idx:
            n_neighbours = total_neighbours[idx]
            if n_neighbours >= least_neighbours:
                continue
            least_neighbours, central_idx = n_neighbours, idx

        if (central_idx is None) or (valid_idx.size == 0):
            break  # There are no clusters

        # Get cluster members
        is_neighbour = similarity_matrix[central_idx].indices

        # Remove central_idx from cluster and append to the end
        is_neighbour = is_neighbour[is_neighbour != central_idx]
        cluster_smiles = np.append(smiles[is_neighbour], smiles[central_idx])
        clusters.append(cluster_smiles.tolist())

        # Remove neighbors of neighbors from the rest of smiles
        nearest_sim = get_similar_mols(smiles, cluster_smiles)
        keep_mask = np.where(nearest_sim < threshold)[0]

        smiles, values = smiles[keep_mask], values[keep_mask]
        similarity_matrix = similarity_matrix[np.ix_(keep_mask, keep_mask)]
        num_mols = len(smiles)

    return clusters, smiles


def lo_train_test_split(
    smiles, threshold, min_cluster_size, max_clusters, values, std_threshold, verbose=False
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
        smiles, threshold, min_cluster_size, max_clusters, 
        values, std_threshold, verbose=verbose
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
