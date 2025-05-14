# Modified from
# https://github.com/isayev/ReLeaSE/blob/master/release/rnn_predictor.py

import pickle
import numpy as np

import torch
import torch.nn as nn

from rdkit import Chem

from models.smiles2label import Smiles2Label
from models.utils import pad_sequences, seq2tensor

class RNNPredictor(nn.Module):
    def __init__(self, path_to_parameters_dict, path_to_checkpoint, tokens):
        super(RNNPredictor, self).__init__()
        self.tokens = ''.join(tokens)
        model_object = Smiles2Label
        model_params = pickle.load(open(path_to_parameters_dict, 'rb'))
        model = []
        for i in range(5):
            model.append(model_object(params=model_params).cuda())
        self.model = model
        self.tokens = tokens

        for i in range(5):
            self.model[i].load_model(path_to_checkpoint + str(i) + '.pkl')

    def predict(self, smiles):
        double = False
        canonical_smiles = []
        invalid_smiles = []

        for i in range(len(smiles)):
            sm = smiles[i]
            try:
                sm = Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=False))
                if len(sm) == 0:
                    invalid_smiles.append(sm)
                else:
                    canonical_smiles.append(sm)
            except:
                invalid_smiles.append(sm)
                
        if len(canonical_smiles) == 0:
            return canonical_smiles, [], invalid_smiles

        if len(canonical_smiles) == 1:
            double = True
            canonical_smiles = [canonical_smiles[0], canonical_smiles[0]]

        padded_smiles, length = pad_sequences(canonical_smiles)
        smiles_tensor, _ = seq2tensor(padded_smiles, self.tokens, flip=False)

        prediction = []
        for i in range(len(self.model)):
            output = self.model[i]([torch.LongTensor(smiles_tensor).cuda(),
                                    torch.LongTensor(length).cuda()],).detach().cpu().numpy()
            prediction.append(np.argmax(output), axis=1)
                
        prediction = np.array([max(l, key=l.count) for l in prediction.T])

        if double:
            canonical_smiles = canonical_smiles[0]
            prediction = [prediction[0]]

        return canonical_smiles, prediction, invalid_smiles