# Modified from
# https://github.com/Mariewelt/OpenChem/blob/master/openchem/models/Smiles2Label.py

from models.base_model import BaseModel

class Smiles2Label(BaseModel):
    def __init__(self, params):
        super(Smiles2Label, self).__init__(params)
        self.embedding = self.params['embedding']
        self.embed_params = self.params['embedding_params']
        self.Embedding = self.embedding(self.embed_params)
        self.encoder = self.params['encoder']
        self.encoder_params = self.params['encoder_params']
        self.Encoder = self.encoder(self.encoder_params, self.use_cuda)
        self.mlp = self.params['mlp']
        self.mlp_params = self.params['mlp_params']
        self.MLP = self.mlp(self.mlp_params)
    
    def forward(self, batch_input):
        input_tensor, input_length = batch_input
        embedded = self.Embedding(input_tensor)
        output, _ = self.Encoder([embedded, input_length])
        output = self.MLP(output)
        return output