import torch
from torch import nn
import torch.nn.functional as F
import config as cfg

class Decoder(nn.Module):
    def __init__(self, wl_size: int, embed_dim: int): # wl_size -> 目标语言的词表大小; embed_dim -> 词嵌入维度
        super().__init__()
        self.embedding = nn.Embedding(wl_size, embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim)
        self.attn_fc = nn.Linear(embed_dim + embed_dim, embed_dim)
        self.new_input_fc = nn.Linear(embed_dim + embed_dim, embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc_final = nn.Linear(embed_dim, wl_size)

    def forward(self, input_word: int, hidden: torch.Tensor, encoder_output: torch.Tensor):
        # hidden: [1, 1, embed_dim]
        # encoder_output: [MAX_SEQ_LENGTH, embed_dim]

        input_word = torch.tensor([input_word], dtype=torch.long, device=cfg.DEVICE)
        embed_input_word = self.embedding(input_word).unsqueeze(0) # [1, 1, embed_dim]
        embed_input_word = embed_input_word.to(cfg.DEVICE)
        hidden = hidden.to(cfg.DEVICE)

        # 本层应当关注的信息
        attn_info = self.attn_fc(torch.cat([embed_input_word[0], hidden[0]], dim=1)) # [1, embed_dim]
        attn_info = self.relu2(attn_info)
        # 本层应当关注的信息与encoder编码的每个词嵌入求相似度
        attn_weight = attn_info @ encoder_output.t() # [1, MAX_SEQ_LENGTH]
        attn_weight = F.softmax(attn_weight, dim=1) # [1, MAX_SEQ_LENGTH]

        attn_ctx = attn_weight @ encoder_output # 需要关注的上下文，即按照注意力权重对编码器输出求加权平均 [1, embed_dim]
        new_input = self.relu(self.new_input_fc(torch.cat([embed_input_word[0], attn_ctx], dim=1))).unsqueeze(0)
        output, hidden = self.gru(new_input, hidden)
        final_output = self.fc_final(output[0])

        return final_output, hidden, attn_weight