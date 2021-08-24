import transformers
import numpy as np
import torch
import torch.nn as nn
import copy
from typing import Optional, Any
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src
        attns = []

        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)

        return output, attns


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        if self.norm_first:
            src = self.norm1(src)
            src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            return src, attn

        # norm last
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn




def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
    
import torch
import torch.nn as nn

import copy
from typing import Optional, Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
import numpy as np



class Lyrics_transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048):
        super(Lyrics_transformer, self).__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("colorfulscoop/gpt2-small-ja")
        self.embedding_layer = nn.Embedding(self.tokenizer.vocab_size, d_model)
        self.positional_encoder = PositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        encoder_layer_norm = LayerNorm(d_model)
        self.encoder = encoder = TransformerEncoder(encoder_layer, num_layers, encoder_layer_norm)
        self.projection_layer = nn.Linear(d_model, self.tokenizer.vocab_size)
    
    def generate_square_subsequent_mask(self, total_len):
        mask = torch.tril(torch.ones(total_len, total_len))  # (total_len, total_len), lower triangle -> 1.; others 0.
        mask = mask == 0  # lower -> False; others True
        return mask
    
    def forward(self, x, gpu):
        mask = self.generate_square_subsequent_mask(x.size(1))
        if gpu:
            mask = mask.cuda()
        x = self.embedding_layer(x)
        x = self.positional_encoder(x)
        x, a = self.encoder(x, mask=mask)
        return self.projection_layer(x), a
    
    def get_train_input(self, df, batch_size, gpu, max_length):
        batch = np.random.choice(len(df), size=(batch_size), replace=False)
        ls = []
        for i in batch:
            ly = df['lyrics'][i].replace('「','').replace('」','').replace('"', '')
            ls.append('<s>'+df['lyrics'][i]+'</s>')
        input = self.tokenizer(ls, truncation=True, padding=True, return_tensors='pt', max_length=max_length)['input_ids']
        if gpu:
            input = input.cuda()
        return input
    
    def train_forward(self, df, batch_size, gpu, max_length):
        input = self.get_train_input(df, batch_size, gpu, max_length)
        output, a = self.forward(input, gpu)
        return {'output':output, 'attention':a, 'input':input}

    def generate(self, gpu, x=None, tmp=0.9, top_k=40, top_p=0, max_len=20):
        with torch.no_grad():
            if x is None:
                x = self.tokenizer('<s>', return_tensors='pt')['input_ids']
            else:
                x = self.tokenizer('<s>'+x, return_tensors='pt')['input_ids']
            if gpu:
                x = x.cuda()
                
            for i in range(max_len):
                output, a = self.forward(x, gpu)
                output = output[0][-1].view(1, 1, self.tokenizer.vocab_size)
                output = sampling_next_token_pk(tmp, output, top_k, top_p).view(1, 1)
                x = torch.cat([x, output], dim=-1)
                if output == self.tokenizer.eos_token_id:
                    break
            
            return self.tokenizer.decode(x.cpu().detach().numpy()[0]), a
    
    def generate_df(self, df, data_num, gpu, max_length, tmp=0.9, top_k=40, top_p=0, max_len=20):
        sp = np.random.choice(['<', '>', '{'])
        output, a = model.generate(gpu, df['lyrics'][data_num].split(sp)[0]+sp, tmp=0.9, top_k=40, top_p=0, max_len=20)
        return {'gt':df['lyrics'][data_num], 'ge':output}

import math
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        
        
import torch.nn.functional as F
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    
    
    return logits

def sampling_next_token_pk(temperature, logits, top_k=0, top_p=0.0):


  # Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
  logits = logits[0, -1, :] / temperature
  filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

  # Sample from the filtered distribution
  probabilities = F.softmax(filtered_logits, dim=-1)
  
  next_token = torch.multinomial(probabilities, 1)

  return next_token

