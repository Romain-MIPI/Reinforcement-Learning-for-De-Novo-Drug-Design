import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem
from models.utils import mol2alt_sentence, sentences2vec

class Mol2VecPredictor():
    def __init__(self, path_to_checkpoint, number_of_fold):
        self.models = []
        self.number_of_fold = number_of_fold
        for i in range(number_of_fold):
            with open(path_to_checkpoint+f'{i}.pkl', 'rb') as f:
                self.models.append(pickle.load(f))

    def predict(self, wv, batch_input):
        batch_input = np.array(batch_input)
        prediction = []
        valid_smiles = []
        invalid_smiles = []

        # get all valid smiles
        ind_valid = []
        mols = []
        for i in range(len(batch_input)):
            m = Chem.MolFromSmiles(batch_input[i])
            if m is not None:
                ind_valid.append(i)
                mols.append(m)

        # embedding to mol2vec
        sentences = [mol2alt_sentence(mol, 1) for mol in mols]
        mol_vec = [x for x in sentences2vec(sentences, wv, unseen='UNK')]

        # predict labels for valid smiles
        tmp = []
        for model in self.models:
            tmp.append(model.predict(mol_vec))
        tmp = np.array(tmp).T
        for i in range(len(tmp)):
            value, count = np.unique(tmp[i], return_counts=True)
            prediction.append(value[np.argmax(count)])

        # collect all prediction
        all_prediction = []
        counter = 0
        for i in range(len(batch_input)):
            if i in ind_valid:
                all_prediction.append(prediction[counter])
                valid_smiles.append(batch_input[i])
                counter += 1
            else:
                all_prediction.append(-1)
                invalid_smiles.append(batch_input[i])

        return valid_smiles, all_prediction, invalid_smiles
        
    def partial_fit(self, new_batch_input, new_batch_label):
        kf = StratifiedKFold(n_splits=self.number_of_fold, shuffle=True, random_state=42)
        for train, _ in kf(new_batch_input, new_batch_label):
            for i in range(self.number_of_fold):
                classes = np.unique(new_batch_label[train])
                self.models[i].partial_fit(new_batch_input[train], new_batch_label[train], classes)

    def save_model(self, path_to_checkpoint):
        for i in range(self.number_of_fold):
            with open(path_to_checkpoint+f'{i}.pkl', 'wb') as f:
                pickle.dump(self.models[i], f)
