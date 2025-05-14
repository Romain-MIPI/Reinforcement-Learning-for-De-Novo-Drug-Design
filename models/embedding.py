from torch import nn

class Embedding(nn.Module):
    def __init__(self, params):
        super(Embedding, self).__init__()
        self.params = params
        self.num_embeddings = self.params['num_embeddings']
        if 'padding_idx' in params.keys():
            self.padding_idx = self.params['padding_idx']
        else:
            self.padding_idx = None
        self.embedding_dim = self.params['embedding_dim']
        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings,
                                      embedding_dim=self.embedding_dim,
                                      padding_idx=self.padding_idx)

    def forward(self, inp):
        embedded = self.embedding(inp)
        return embedded