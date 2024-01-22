import numpy as np

def positional_encoding(seq_length, embedding_dim):
    position = np.arange(seq_length)[: , np.newaxis]
    div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
    # 计算正余弦函数的值
    pos_enc = np.zeros((seq_length, embedding_dim))
    pos_enc[:, 0::2] = np.sin(position * div_term)

    return pos_enc
