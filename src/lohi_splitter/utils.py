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

    lhs_mols = []
    for smiles in lhs:
        lhs_mols.append(Chem.MolFromSmiles(smiles))
    lhs_fps = [fp_generator.GetFingerprint(x) for x in lhs_mols]

    rhs_mols = []
    for smiles in rhs:
        rhs_mols.append(Chem.MolFromSmiles(smiles))
    rhs_fps = [fp_generator.GetFingerprint(x) for x in rhs_mols]

    nearest_sim = []
    nearest_idx = []
    for lhs in lhs_fps:
        sims = DataStructs.BulkTanimotoSimilarity(lhs, rhs_fps)
        nearest_sim.append(max(sims))
        nearest_idx.append(np.argmax(sims))
    if return_idx:
        result = (nearest_sim, nearest_idx)
    else:
        result = nearest_sim
    return result
