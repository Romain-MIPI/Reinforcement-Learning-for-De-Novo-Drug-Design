import warnings
import numpy as np
import csv
import time
import math
from rdkit import Chem
from rdkit.Chem import AllChem

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

def canonical_smiles(smiles, sanitize=True, throw_warning=False):
    """
    Takes list of SMILES strings and returns list of their canonical SMILES.

    Parameters
    ----------
    smiles: list
        list of SMILES strings to convert into canonical format

    sanitize: bool (default True)
        parameter specifying whether to sanitize SMILES or not.
            For definition of sanitized SMILES check
            http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol

    throw_warning: bool (default False)
        parameter specifying whether warnings will be thrown if a SMILES is
        invalid

    Returns
    -------
    new_smiles: list
        list of canonical SMILES and NaNs if SMILES string is invalid or
        unsanitized (when sanitize is True)

    When sanitize is True the function is analogous to:
        sanitize_smiles(smiles, canonical=True).
    """
    new_smiles = []
    for sm in smiles:
        try:
            mol = Chem.MolFromSmiles(sm, sanitize=sanitize)
            new_smiles.append(Chem.MolToSmiles(mol))
        except:
            if throw_warning:
                warnings.warn(sm + ' can not be canonized: invalid '
                                   'SMILES string!', UserWarning)
            new_smiles.append('')
    return new_smiles

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

def read_smi_file(filename, unique=True, add_start_end_tokens=False):
    """
    Reads SMILES from file. File must contain one SMILES string per line
    with \n token in the end of the line.

    Args:
        filename (str): path to the file
        unique (bool): return only unique SMILES

    Returns:
        smiles (list): list of SMILES strings from specified file.
        success (bool): defines whether operation was successfully completed or not.

    If 'unique=True' this list contains only unique copies.
    """
    f = open(filename, 'r')
    molecules = []
    for line in f:
        if add_start_end_tokens:
            molecules.append('<' + line[:-1] + '>')
        else:
            molecules.append(line[:-1])
    if unique:
        molecules = list(set(molecules))
    else:
        molecules = list(molecules)
    f.close()
    return molecules, f.closed

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

def read_object_property_file(path, delimiter=',', cols_to_read=[0, 1], keep_header=False):
    f = open(path, 'r')
    reader = csv.reader(f, delimiter=delimiter)
    data_full = np.array(list(reader))
    if keep_header:
        start_position = 0
    else:
        start_position = 1
    assert len(data_full) > start_position
    data = [[] for _ in range(len(cols_to_read))]
    for i in range(len(cols_to_read)):
        col = cols_to_read[i]
        data[i] = data_full[start_position:, col]
    f.close()
    if len(cols_to_read) == 1:
        data = data[0]
    return data

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

def sentences2vec(sentences, model, unseen=None):
    """Generate vectors for each sentence (list) in a list of sentences. Vector is simply a
    sum of vectors for individual words.
    
    Parameters
    ----------
    sentences : list, array
        List with sentences
    model : word2vec.Word2Vec
        Gensim word2vec model
    unseen : None, str
        Keyword for unseen words. If None, those words are skipped.
        https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032

    Returns
    -------
    np.array
    """
    keys = set(model.wv.vocab.keys())
    vec = []
    if unseen:
        unseen_vec = model.wv.word_vec(unseen)

    for sentence in sentences:
        if unseen:
            vec.append(sum([model.wv.word_vec(y) if y in set(sentence) & keys
                       else unseen_vec for y in sentence]))
        else:
            vec.append(sum([model.wv.word_vec(y) for y in sentence 
                            if y in set(sentence) & keys]))

    print("###############################\n")        
    for i in range(len(vec)):
        print(len(vec[i]))
    print("###############################\n") 

    return np.array(vec)

def mol2alt_sentence(mol, radius):
    """Same as mol2sentence() expect it only returns the alternating sentence
    Calculates ECFP (Morgan fingerprint) and returns identifiers of substructures as 'sentence' (string).
    Returns a tuple with 1) a list with sentence for each radius and 2) a sentence with identifiers from all radii
    combined.
    NOTE: Words are ALWAYS reordered according to atom order in the input mol object.
    NOTE: Due to the way how Morgan FPs are generated, number of identifiers at each radius is smaller
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float 
        Fingerprint radius
    
    Returns
    -------
    list
        alternating sentence
    combined
    """
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)