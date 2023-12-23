import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size,device):
        super(GRUCell, self).__init__()
        self.W_xr = nn.Linear(input_size, hidden_size).to(device)
        self.W_hr = nn.Linear(hidden_size, hidden_size).to(device)
        self.W_xz = nn.Linear(input_size, hidden_size).to(device)
        self.W_hz = nn.Linear(hidden_size, hidden_size).to(device)
        self.W_xh = nn.Linear(input_size, hidden_size).to(device)
        self.W_hh = nn.Linear(hidden_size, hidden_size).to(device)

    def forward(self, X_cur, H_pre):
        Rt = torch.sigmoid(self.W_xr(X_cur) + self.W_hr(H_pre))
        Zt = torch.sigmoid(self.W_xz(X_cur) + self.W_hz(H_pre))
        H_candidate = torch.tanh(self.W_xh(X_cur) + self.W_hh(Rt * H_pre))
        H = Zt * H_pre + (1 - Zt) * H_candidate
        return H


class GRU(nn.Module):  # time_series encoder
    def __init__(self, num_periods, input_size, hidden_size,device):
        super(GRU, self).__init__()
        self.device = device
        self.num_periods = num_periods
        self.hidden_size = hidden_size
        self.TSEncoder = nn.ModuleList([GRUCell(input_size, hidden_size,device) for _ in range(num_periods)])

    def forward(self, X):
        """
        :param X: shape: (batch_size, num_periods, num_nodes, time_length, input_size
        :return:
        """
        B, P, N, T, C = X.shape
        # print(X.shape)
        H = torch.zeros(B, N, T, self.hidden_size).to(self.device)
        for day in range(self.num_periods):
            X_piece = X[:, day]
            H = self.TSEncoder[day](X_piece, H)

        return H


class ABCGRU_layer(nn.Module):  # spatial attention GRU
    def __init__(self, input_size, hidden_size, device):
        super(ABCGRU_layer, self).__init__()
        self.W_xr = nn.Linear(input_size, hidden_size).to(device)
        self.W_hr = nn.Linear(hidden_size, hidden_size).to(device)
        self.W_xz = nn.Linear(input_size, hidden_size).to(device)
        self.W_hz = nn.Linear(hidden_size, hidden_size).to(device)
        self.W_xh = nn.Linear(input_size, hidden_size).to(device)
        self.W_hh = nn.Linear(hidden_size, hidden_size).to(device)

        self.W_ha1 = nn.Linear(input_size, hidden_size, bias=False).to(device)
        self.W_ha2 = nn.Linear(hidden_size, hidden_size).to(device)

        self.W_y = nn.Linear(hidden_size, input_size).to(device)

    def forward(self, H_pre, X_cur):
        B, N, C = X_cur.shape
        lhs = self.W_ha1(X_cur)  # (B, N, C) * (C,C') -> (B,N,C')
        rhs = self.W_ha2(H_pre).transpose(-1, -2)  # (B, N, C') -> (B,N, C') -> (B, C', N)
        S = F.leaky_relu(torch.matmul(lhs, rhs))
        S_norm = F.softmax(S, dim=-1)
        Rt = torch.sigmoid(self.W_xr(S_norm @ X_cur))
        Zt = torch.sigmoid(self.W_xz(S_norm @ X_cur))
        H_candidate = torch.tanh(self.W_xh(X_cur) + self.W_hh(Rt * H_pre))
        H = Zt * H_pre + (1 - Zt) * H_candidate
        y = self.W_y(H)

        # Rt = torch.sigmoid(self.W_xr(X_cur) + self.W_hr(H_pre))
        # Zt = torch.sigmoid(self.W_xz(X_cur) + self.W_hz(H_pre))
        # H_candidate = torch.tanh(self.W_xh(X_cur) + self.W_hh(Rt * H_pre))
        # H = Zt * H_pre + (1 - Zt) * H_candidate
        # y = self.W_y(H)

        return y, H


class ABCGRU_encoder(nn.Module):
    def __init__(self, time_length, input_size, hidden_size,device):
        super(ABCGRU_encoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.time_length = time_length
        self.encoder = nn.ModuleList([ABCGRU_layer(input_size, hidden_size, device) for _ in range(time_length)])

    def forward(self, X):
        B, N, T, C = X.shape
        H = torch.zeros(B, N, self.hidden_size).to(self.device)
        for time_step in range(self.time_length):
            X_piece = X[:, :, time_step]
            _, H = self.encoder[time_step](H, X_piece)

        return H


class ABCGRU_decoder(nn.Module):
    def __init__(self, pre_length, input_size, hidden_size,device):
        super(ABCGRU_decoder, self).__init__()
        self.device = device
        self.decoder = nn.ModuleList([ABCGRU_layer(input_size, hidden_size,device) for _ in range(pre_length)])
        self.input_size = input_size
        self.pre_length = pre_length

    def forward(self, X, H):
        B, N, T, C = X.shape
        output = torch.zeros(B, N, C).to(self.device)
        outputs = []
        for time_step in range(self.pre_length):
            output, H = self.decoder[time_step](H, output)
            outputs.append(output.unsqueeze(-1))

        ans = torch.concat(outputs, dim=-1)
        return ans.permute(0, 1, -1, -2)


class ABCGRU(nn.Module):
    def __init__(self, input_size, output_size, time_length, pre_length,device):
        super(ABCGRU, self).__init__()
        self.encoder = ABCGRU_encoder(time_length, input_size, output_size,device)
        self.decoder = ABCGRU_decoder(pre_length, input_size, output_size,device)

    def forward(self, X):
        H = self.encoder(X)
        ans = self.decoder(X, H)
        return ans




class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] 可能没有
        '''
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # scores : [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), N(Spatial) or T(Temporal)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,
                               V)  # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context


class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads,device):
        super(TMultiHeadAttention, self).__init__()

       # self.K_size = K_size
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
       # self.head_dim2 = K_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False).to(device)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False).to(device)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False).to(device)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size).to(device)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output

def positional_encoding(inputs, device):
    batch_size, nodes, time_steps, features = inputs.size()
 
    pos_enc = torch.zeros((time_steps, features)).to(device)
    for i in range(time_steps):
        for j in range(features):
            if j % 2 == 0:
                pos_enc[i, j] = torch.sin(torch.FloatTensor([i / 10000**(2 * j / features)]))
            else:
                pos_enc[i, j] = torch.cos(torch.FloatTensor([i / 10000**(2 * j / features)]))
    
    pos_enc = pos_enc.unsqueeze(0).unsqueeze(0)
 
    inputs_with_pos_enc = inputs + pos_enc
    
    return inputs_with_pos_enc

class My_model(nn.Module):
    def __init__(self, input_size, hidden_size, time_length, pre_length, num_day, num_week, heads,device):
        super(My_model, self).__init__()
        self.line0r = nn.Linear(1, input_size).to(device)
        self.line0d = nn.Linear(1, input_size).to(device)
        self.line0w = nn.Linear(1, input_size).to(device)

        # self.line1r = nn.Linear(input_size, input_size*2).to(device)
        # self.line1d = nn.Linear(input_size, input_size*2).to(device)
        # self.line1w = nn.Linear(input_size, input_size*2).to(device)

        

        self.ABCG = ABCGRU(input_size, hidden_size, time_length, pre_length,device)
        self.liner = nn.Linear(input_size, hidden_size).to(device)

        self.GRU1 = GRU(num_day, input_size, hidden_size,device)
        self.GRU2 = GRU(num_week, input_size, hidden_size,device)
        self.TMHAtt1 = TMultiHeadAttention(hidden_size, heads,device)
        self.TMHAtt2 = TMultiHeadAttention(hidden_size, heads,device)
        self.LN1 = nn.LayerNorm(hidden_size).to(device)
        self.LN2 = nn.LayerNorm(hidden_size).to(device)

        self.line1_1 = nn.Linear(hidden_size, hidden_size*2).to(device)
        self.line1_2 = nn.Linear(hidden_size*2, hidden_size).to(device)
        self.line2_1 = nn.Linear(hidden_size, hidden_size*2).to(device)
        self.line2_2 = nn.Linear(hidden_size*2, hidden_size).to(device)

        self.line3_1 = nn.Linear(hidden_size, 16).to(device)
        self.line3_1_1 = nn.Linear(16, 8).to(device)
        self.line3_2 = nn.Linear(8, 1).to(device)
        self.device = device

    def forward(self, Xr, Xd, Xw):
        Xr = F.relu(self.line0r(Xr))
        Xd = F.relu(self.line0d(Xd))
        Xw = F.relu(self.line0w(Xw))

        y_candidate = self.ABCG(Xr)
        y_candidate = torch.relu(self.liner(y_candidate))
        y_candidate = positional_encoding(y_candidate, self.device)

        Hd = self.GRU1(Xd)
        Hd = positional_encoding(Hd, self.device)
        # Hw = self.GRU2(Xw)
        # Hw = positional_encoding(Hw, self.device)

        y_1 = self.TMHAtt1(y_candidate, Hd, Hd)
        y_1 = self.LN1(y_candidate + y_1)
        y_1 = self.line1_2(F.relu(self.line1_1(y_1)))

        # y_1 = positional_encoding(y_1,self.device)

        # y_2 = self.TMHAtt2(y_1, Hw, Hw)
        # y_2 = self.LN2(y_1 + y_2)
        # y_2 = self.line2_2(F.relu(self.line2_1(y_2)))

        # print(y_2.shape)
        y_hat = self.line3_2(F.relu(self.line3_1_1(F.relu(self.line3_1(y_1)))))
        return y_hat[..., 0].squeeze(-1)






