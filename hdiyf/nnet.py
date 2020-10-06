import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    '''
    Attention Layer
    '''
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        '''
        :param x: (batch_size, max_len, hidden_size)
        :return alpha: (batch_size, max_len)
        '''
        x = torch.tanh(x)  # (batch_size, max_len, hidden_size)
        x = self.attn(x).squeeze(2)  # (batch_size, max_len)
        alpha = F.softmax(x, dim=1).unsqueeze(1)  # (batch_size, 1, max_len)
        return alpha


class AttentiveBilstm(nn.Module):

    def __init__(self, embeddings, args):
        super().__init__()

        num_class = args['class_num']

        self.word_embed = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings), freeze=True)

        total_embedding_dim = args['word_embedding_dim']

        self.embed_dropout = nn.Dropout(args['fc_dropout'])

        self.contextual_hidden_dim = args["contextual_hidden_dim"]

        self.bilstm = nn.LSTM(total_embedding_dim,
                              self.contextual_hidden_dim,
                              num_layers=args["num_layers"],
                              bidirectional=True,
                              batch_first=True)

        self.fc_dropout = nn.Dropout(args['fc_dropout'])
        self.fc = nn.Linear(self.contextual_hidden_dim, num_class)
        self.attention = Attention(self.contextual_hidden_dim)

    def forward(self, x, y=None):
        text_embedding = self.word_embed(x)
        text_contextual, _ = self.bilstm(text_embedding)  # (batch_size, max_len, hidden_size*num_directions)
        text_contextual = text_contextual[:, :, :self.contextual_hidden_dim] + \
                          text_contextual[:, :, self.contextual_hidden_dim:]  # (batch_size, max_len, hidden_size)
        context_vector = torch.mean(text_contextual, dim=1)  # (batch_size, total_embedding_dim)

        alpha = self.attention(text_contextual)  # (batch_size, 1, max_len)
        r = alpha.bmm(text_contextual).squeeze(1)  # (batch_size, hidden_size)
        h = torch.tanh(r)  # (batch_size, hidden_size)
        #cat = torch.cat([context_vector, h], dim=-1)
        logits = self.fc_dropout(self.fc(h))  # (batch_size, class_num)

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        return logits, loss