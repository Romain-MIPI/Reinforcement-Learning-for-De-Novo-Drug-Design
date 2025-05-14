import warnings
import numpy as np
import csv
import time
import math
from rdkit import Chem

from models.smiles_enumerator import SmilesEnumerator

def get_tokens(smiles, tokens=None):
    """
    Returns list of unique tokens, token-2-index dictionary and
    number of unique tokens from the list of SMILES
    Args:
        smiles (list): list of SMILES strings to tokenize.
        tokens (string): string of tokens or None.
        If none will be extracted from dataset.
    Returns:
        tokens (list): list of unique tokens/SMILES alphabet.
        token2idx (dict): dictionary mapping token to its index.
        num_tokens (int): number of unique tokens.
    """
    if tokens is None:
        tokens = list(set(''.join(smiles)))
        tokens = sorted(tokens)
        tokens = ''.join(tokens)
    token2idx = dict((token, i) for i, token in enumerate(tokens))
    num_tokens = len(tokens)
    return tokens, token2idx, num_tokens

def augment_smiles(smiles, labels, n_augment=5):
    smiles_augmentation = SmilesEnumerator()
    augmented_smiles = []
    augmented_labels = []
    for i in range(len(smiles)):
        sm = smiles[i]
        for _ in range(n_augment):
            augmented_smiles.append(smiles_augmentation.randomize_smiles(sm))
            augmented_labels.append(labels[i])
        augmented_smiles.append(sm)
        augmented_labels.append(labels[i])
    return augmented_smiles, augmented_labels

def sanitize_smiles(smiles,
                    canonize=True,
                    min_atoms=-1,
                    max_atoms=-1,
                    return_num_atoms=False,
                    allowed_tokens=None,
                    allow_charges=False,
                    return_max_len=False,
                    logging="warn"):
    """
    Takes list of SMILES strings and returns list of their sanitized versions.
    For definition of sanitized SMILES check
    http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol
        Args:
            smiles (list): list of SMILES strings
            canonize (bool): parameter specifying whether to return
            canonical SMILES or not.
            min_atoms (int): minimum allowed number of atoms
            max_atoms (int): maxumum allowed number of atoms
            return_num_atoms (bool): return additional array of atom numbers
            allowed_tokens (iterable, optional): allowed tokens set
            allow_charges (bool): allow nonzero charges of atoms
            logging ("warn", "info", "none"): logging level
        Output:
            new_smiles (list): list of SMILES and NaNs if SMILES string is
            invalid or unsanitized.
            If 'canonize = True', return list of canonical SMILES.
        When 'canonize = True' the function is analogous to:
        canonize_smiles(smiles, sanitize=True).
    """
    assert logging in ["warn", "info", "none"]

    new_smiles = []
    idx = []
    num_atoms = []
    smiles_lens = []
    for i in range(len(smiles)):
        sm = smiles[i]
        mol = Chem.MolFromSmiles(sm, sanitize=False)
        sm_new = Chem.MolToSmiles(mol) if canonize and mol is not None else sm
        good = mol is not None
        if good and allowed_tokens is not None:
            good &= all([t in allowed_tokens for t in sm_new])

        if good and not allow_charges:
            good &= all([a.GetFormalCharge() == 0 for a in mol.GetAtoms()])

        if good:
            n = mol.GetNumAtoms()
            if (n < min_atoms and min_atoms > -1) or (n > max_atoms > -1):
                good = False
        else:
            n = 0

        if good:
            new_smiles.append(sm_new)
            idx.append(i)
            num_atoms.append(n)
            smiles_lens.append(len(sm_new))
        else:
            new_smiles.append(' ')
            num_atoms.append(0)

    smiles_set = set(new_smiles)
    num_unique = len(smiles_set) - ('' in smiles_set)
    if len(idx) > 0:
        valid_unique_rate = float(num_unique) / len(idx)
        invalid_rate = 1.0 - float(len(idx)) / len(smiles)
    else:
        valid_unique_rate = 0.0
        invalid_rate = 1.0
    num_bad = len(smiles) - len(idx)

    if len(idx) != len(smiles) and logging == "warn":
        warnings.warn('{:d}/{:d} unsanitized smiles ({:.1f}%)'.format(num_bad, len(smiles), 100 * invalid_rate))
    elif logging == "info":
        print("Valid: {}/{} ({:.2f}%)".format(len(idx), len(smiles), 100 * (1 - invalid_rate)))
        print("Unique valid: {:.2f}%".format(100 * valid_unique_rate))

    if return_num_atoms and return_max_len:
        return new_smiles, idx, num_atoms, max(smiles_lens)
    elif return_num_atoms and not return_max_len:
        return new_smiles, idx, num_atoms
    elif not return_num_atoms and return_max_len:
        return new_smiles, idx, max(smiles_lens)
    else:
        return new_smiles, idx

def seq2tensor(seqs, tokens, flip=True):
    tensor = np.zeros((len(seqs), len(seqs[0])))
    for i in range(len(seqs)):
        for j in range(len(seqs[i])):
            if seqs[i][j] in tokens:
                tensor[i, j] = tokens.index(seqs[i][j])
            else:
                tokens = tokens + seqs[i][j]
                tensor[i, j] = tokens.index(seqs[i][j])
    if flip:
        tensor = np.flip(tensor, axis=1).copy()
    return tensor, tokens

def pad_sequences(seqs, max_length=None, pad_symbol=' '):
    if max_length is None:
        max_length = -1
        for seq in seqs:
            max_length = max(max_length, len(seq))
    lengths = []
    for i in range(len(seqs)):
        cur_len = len(seqs[i])
        lengths.append(cur_len)
        seqs[i] = seqs[i] + pad_symbol * (max_length - cur_len)
    return seqs, lengths

def save_smiles_property_file(path, smiles, labels, delimiter=','):
    f = open(path, 'w')
    n_targets = labels.shape[1]
    for i in range(len(smiles)):
        f.writelines(smiles[i])
        for j in range(n_targets):
            f.writelines(delimiter + str(labels[i, j]))
        f.writelines('\n')
    f.close()

def read_smiles_property_file(path, cols_to_read, delimiter=',', keep_header=False):
    reader = csv.reader(open(path, 'r'), delimiter=delimiter)
    data = list(reader)
    if keep_header:
        start_position = 0
    else:
        start_position = 1
    assert len(data) > start_position
    data = map(list, zip(*data))
    data = [d for c, d in enumerate(data)]
    data_ = [data[c][start_position:] for c in cols_to_read]

    return data_

def process_smiles(smiles,
                   sanitized=False,
                   target=None,
                   augment=False,
                   pad=True,
                   tokenize=True,
                   tokens=None,
                   flip=False,
                   allowed_tokens=None):
    if not sanitized:
        clean_smiles, clean_idx = sanitize_smiles(smiles, allowed_tokens=allowed_tokens)
        clean_smiles = [clean_smiles[i] for i in clean_idx]
        if target is not None:
            target = target[clean_idx]
    else:
        clean_smiles = smiles

    length = None
    if augment and target is not None:
        clean_smiles, target = augment_smiles(clean_smiles, target)
    if pad:
        clean_smiles, length = pad_sequences(clean_smiles)
    tokens, token2idx, num_tokens = get_tokens(clean_smiles, tokens)
    if tokenize:
        clean_smiles, tokens = seq2tensor(clean_smiles, tokens, flip)

    return clean_smiles, target, length, tokens, token2idx, num_tokens

def identity(input):
    return input

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)