import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Common import MoEAttenBlock, SelfAttention


class LogitGate(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hidden_dim = input_dim // 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.mlp(x)


class AGMM(nn.Module):
    def __init__(
        self,
        num_item: int,
        d: int = 512,
        d_V: int = 512,
        d_T: int = 512,
        num_encoder_layers: int = 3,
        num_encoder_heads: int = 4,
        num_item_attn_layers: int = 3,
        num_item_attn_heads: int = 4,
        num_gen_layers: int = 2,
        num_gen_heads: int = 3,
        k: int = 3,
        max_len: int = 50,
        hidden_dropout_prob: float = 0.2,
        attn_dropout_prob: float = 0.2,
        modal_dropout_prob: float = 0.2,
        item_dropout_prob: float = 0.2,
        use_positional_embedding: bool = True,
        logit_scale_init_value: float = 2.6592,
        device: str = "cpu",
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__()
        self.num_item = num_item

        self.d = d
        self.d_V = d_V
        self.d_T = d_T

        self.num_encoder_layers = num_encoder_layers
        self.num_encoder_heads = num_encoder_heads
        self.num_item_attn_layers = num_item_attn_layers
        self.num_item_attn_heads = num_item_attn_heads
        self.num_gen_layers = num_gen_layers
        self.num_gen_heads = num_gen_heads

        self.use_positional_embedding = use_positional_embedding
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.k = k

        self.initializer_range = initializer_range
        self.device = device

        self.item_emb = nn.Embedding(num_item + 1, self.d, padding_idx=0)
        self.text_emb = nn.Embedding(num_item + 1, self.d_T, padding_idx=0)
        self.img_emb = nn.Embedding(num_item + 1, self.d_V, padding_idx=0)

        self.m_dropout = nn.Dropout(modal_dropout_prob)
        self.i_dropout = nn.Dropout(item_dropout_prob)
        self.emb_layernorm = nn.LayerNorm(d, eps=1e-6)

        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))

        if self.use_positional_embedding:
            self.positional_emb = nn.Embedding(max_len + 1, self.d, padding_idx=0)

        ########## Generation Module
        self.d_T_to_d = nn.Linear(self.d_T, self.d)
        self.d_V_to_d = nn.Linear(self.d_V, self.d)
        self.d_to_d_T = nn.Linear(self.d, self.d_T)
        self.d_to_d_V = nn.Linear(self.d, self.d_V)

        self.item_imp = LogitGate(self.d)
        self.text_imp = LogitGate(self.d_T)
        self.img_imp = LogitGate(self.d_V)

        self.text_generator = nn.ModuleList(
            [
                SelfAttention(
                    self.num_gen_heads,
                    self.d,
                    self.hidden_dropout_prob,
                    self.attn_dropout_prob,
                )
                for _ in range(num_gen_layers)
            ]
        )

        self.img_generator = nn.ModuleList(
            [
                SelfAttention(
                    self.num_gen_heads,
                    self.d,
                    self.hidden_dropout_prob,
                    self.attn_dropout_prob,
                )
                for _ in range(num_gen_layers)
            ]
        )

        ########## Context Encoder
        self.item_SA = nn.ModuleList(
            [
                SelfAttention(
                    self.num_encoder_heads,
                    self.d,
                    self.hidden_dropout_prob,
                    self.attn_dropout_prob,
                )
                for _ in range(self.num_item_attn_layers)
            ]
        )

        self.encoder = nn.ModuleList(
            [
                MoEAttenBlock(
                    self.num_encoder_heads,
                    self.d,
                    self.hidden_dropout_prob,
                    self.attn_dropout_prob,
                    self.k,
                )
                for _ in range(self.num_encoder_layers)
            ]
        )

        # ########## Adaptive Gating
        self.v2d = nn.Linear(self.d_V, self.d)
        self.t2d = nn.Linear(self.d_T, self.d)

        self.adaptive_gate_KV = nn.Sequential(
            nn.Linear(self.d * 2, self.d),
            nn.ReLU(),
            nn.Linear(self.d, 2),
        )
        self.adaptive_gate_Q = nn.Sequential(
            nn.Linear(self.d * 2, self.d),
            nn.ReLU(),
            nn.Linear(self.d, 2),
        )

        ########## Parameter Init
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.fill_(0.5)

    def get_attention_mask(self, item_seq, bidirectional=False):
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000)
        return extended_attention_mask

    def forward(self, seqs, S_V, S_T):
        S_V = S_V.to(self.device)
        S_T = S_T.to(self.device)
        S_I = self.item_emb(seqs).to(self.device)

        ########## MASK && Positional Eembedding
        key_padding_mask = seqs == 0
        extended_attention_mask = (
            self.get_attention_mask(seqs).squeeze().repeat(2, 1, 1)
        )

        S_I = self.i_dropout(self.emb_layernorm(S_I))
        if self.use_positional_embedding:
            positions = np.tile(np.array(range(seqs.shape[1])), [seqs.shape[0], 1])
            positions = torch.tensor(positions).to(self.device)
            positions[seqs == 0] = 0
            pos_emb = self.positional_emb(positions)
            S_I = S_I + pos_emb

        S_I = S_I.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        S_T = S_T.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        S_V = S_V.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        ########## Generation Module
        G = self.d_T_to_d(self.m_dropout(S_T)) + S_I
        G = G.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        for enc in self.text_generator:
            G = enc(G, extended_attention_mask)
            G = G.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        T_L = self.d_to_d_T(G)
        T_L = T_L.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        t_hat = G
        t_hat = t_hat.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        G_V = self.d_V_to_d(self.m_dropout(S_V)) + S_I
        G_V = G_V.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        for enc in self.img_generator:
            G_V = enc(G_V, extended_attention_mask)
            G_V = G_V.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        V_L = self.d_to_d_V(G_V)
        V_L = V_L.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        v_hat = G_V
        v_hat = v_hat.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        ########## Adaptive Gating
        S_V = self.v2d(S_V)
        S_T = self.t2d(S_T)
        S_V = S_V.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        S_T = S_T.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        S_V = F.layer_norm(S_V, (self.d,))
        S_T = F.layer_norm(S_T, (self.d,))
        S_I = F.layer_norm(S_I, (self.d,))

        S_V = S_V.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        S_T = S_T.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        S_I = S_I.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        if self.is_gate:
            modal_features_concat = torch.cat([S_V, S_T], dim=-1)
            gate_logits = self.adaptive_gate_KV(modal_features_concat)
            alpha = torch.sigmoid(gate_logits / self.temp)
            m = S_V * alpha[..., 0].unsqueeze(-1) + S_T * alpha[..., 1].unsqueeze(-1)
        else:
            m = S_V + S_T

        m = m.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        KV_R = S_I + m

        i_hat = S_I
        for block in self.item_SA:
            i_hat = block(i_hat, extended_attention_mask)
            i_hat = i_hat.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        v_hat = F.layer_norm(v_hat, (self.d,))
        t_hat = F.layer_norm(t_hat, (self.d,))
        i_hat = F.layer_norm(i_hat, (self.d,))

        v_hat = v_hat.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        t_hat = t_hat.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        i_hat = i_hat.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        if self.is_gate:
            generated_features_concat_q = torch.cat([v_hat, t_hat], dim=-1)
            gate_logits_q = self.adaptive_gate_Q(generated_features_concat_q)
            alpha_g = torch.sigmoid(gate_logits_q / self.temp)
            m_g = v_hat * alpha_g[..., 0].unsqueeze(-1) + t_hat * alpha_g[
                ..., 1
            ].unsqueeze(-1)
        else:
            m_g = v_hat + t_hat

        m_g = m_g.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        Q_R = i_hat + m_g

        for block in self.encoder:
            Q_R = block(KV_R, Q_R, extended_attention_mask)
            Q_R = Q_R.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        Q_R = Q_R.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        V_L = V_L.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        T_L = T_L.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        logits_item = torch.matmul(Q_R, self.item_emb.weight.T)
        logits_text = torch.matmul(T_L, self.text_emb.weight.detach().T)
        logits_img = torch.matmul(V_L, self.img_emb.weight.detach().T)

        i_imps = torch.sigmoid(self.item_imp(self.item_emb.weight).squeeze())
        t_imps = torch.sigmoid(
            self.text_imp(self.text_emb.weight.detach()).squeeze()
        )
        v_imps = torch.sigmoid(
            self.img_imp(self.img_emb.weight.detach()).squeeze()
        )

        logits = (
            (logits_item * i_imps) + (logits_text * t_imps) + (logits_img * v_imps)
        )

        return (
            V_L,
            T_L,
            logits
        )
