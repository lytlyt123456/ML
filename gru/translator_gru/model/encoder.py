import torch
from torch import nn
import config as cfg

class Encoder(nn.Module):
    def __init__(self, wl_size: int, embed_dim: int): # wl_size -> 待翻译语言的词表大小; embed_dim -> 词嵌入维度
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(wl_size, embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim)

    def forward(self, input_word: int, hidden: torch.Tensor):
        if hidden is None:
            hidden = torch.zeros([1, 1, self.embed_dim])
        input_word = torch.tensor([input_word], dtype=torch.long, device=cfg.DEVICE)
        hidden = hidden.to(cfg.DEVICE)
        embed_input_word = self.embedding(input_word) # [1, embed_dim]
        embed_input_word = embed_input_word.unsqueeze(0) # [1, 1, embed_dim]
        output, hidden = self.gru(embed_input_word, hidden)
        return output, hidden