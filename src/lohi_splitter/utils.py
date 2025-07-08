from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np


def get_similar_mols(lhs, rhs, return_idx=False):
    """
    Calculated maximal similarity between two sets for each molecule.

    Parameters:
        lhs -- list of smiles
        rhs -- list of smiles
        return_idx -- if True also returns idx of the similar molecules

    Returns:
        if return_idx = False:
            nearest_sim -- list of length of lhs. i'th element contains maximal similarity between lhs[i] and rhs
        if return_idx = True:
            (nearest_sim, nearest_idx)
            nearest_idx -- list of length of lhs. i'th element contains idx of rhs molecule, which is similar to lhs[i]
    """
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    # Convert SMILES to fingerprints
    lhs_fps = [fp_generator.GetFingerprint(Chem.MolFromSmiles(smile)) for smile in lhs]
    rhs_fps = [fp_generator.GetFingerprint(Chem.MolFromSmiles(smile)) for smile in rhs]

    # Compute similarities in bulk
    nearest_sim = np.zeros(len(lhs))
    nearest_idx = np.zeros(len(lhs), dtype=int)

    for i, lhs_fp in enumerate(lhs_fps):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(lhs_fp, rhs_fps))
        nearest_idx[i] = sims.argmax()
        nearest_sim[i] = sims[nearest_idx[i]]

    return (nearest_sim, nearest_idx) if return_idx else nearest_sim
