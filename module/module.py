import numpy as np
import torch
import torch.nn as nn
from tools.attention import Attention
import torch.nn.functional as F
import module.ws as ws

class ObjectLanguageAlignmentLayer(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        
        # attention
        self.obj_attn = Attention(embed_dim=configs["lang_token_dim"], num_heads=8, weight_standardization=False)
        self.motion_attn = Attention(embed_dim=configs["lang_token_dim"], num_heads=8, weight_standardization=False)
        self.object2lang_attn = Attention(embed_dim=configs["lang_token_dim"], num_heads=8, weight_standardization=False)
        
        # normalization
        self.norm = nn.ModuleList([
            nn.GroupNorm(num_groups=configs["n_groups_module"], num_channels=configs["lang_token_dim"]) for _ in range(3)
        ])

    def forward(
        self,
        object_tokens,
        object_tokens_pe,
        lang_tokens,
    ):
        b, n_obj, t, d = object_tokens.shape
        
        # inter-object attention
        object_tokens = object_tokens.permute(0, 2, 1, 3).reshape(b * t, n_obj, d)
        inter_obj_attn_out = self.obj_attn(q=object_tokens, k=object_tokens, v=object_tokens)
        object_tokens = object_tokens + inter_obj_attn_out
        object_tokens = self.norm[0](object_tokens.permute(0, 2, 1)).permute(0, 2, 1)
        object_tokens = object_tokens.reshape(b, t, n_obj, d).permute(0, 2, 1, 3)
        
        # motion attention
        object_tokens_ = object_tokens + object_tokens_pe
        object_tokens = object_tokens.reshape(b * n_obj, t, d)
        object_tokens_ = object_tokens_.reshape(b * n_obj, t, d)
        motion_attn_out = self.motion_attn(q=object_tokens_, k=object_tokens_, v=object_tokens)
        object_tokens = object_tokens + motion_attn_out
        object_tokens = self.norm[1](object_tokens.permute(0, 2, 1)).permute(0, 2, 1)
        
        # object to language attention
        object_tokens = object_tokens.reshape(b, n_obj * t, d)
        object2lang_attn_out = self.object2lang_attn(q=object_tokens, k=lang_tokens, v=lang_tokens)
        object_tokens = object_tokens + object2lang_attn_out
        object_tokens = self.norm[2](object_tokens.permute(0, 2, 1)).permute(0, 2, 1)
        object_tokens = object_tokens.reshape(b, n_obj, t, d)

        return object_tokens, lang_tokens

class LanguageAlignedTrackSelectionModule(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        
        # configs
        self.object_token_dim = configs["object_token_dim"]
        self.lang_token_dim = configs["lang_token_dim"]
        self.n_layers = configs["n_layers"]
        self.max_temporal_length = configs["max_temporal_length"]
        self.n_negative = configs["n_negative"]
        
        # short-term motion encoder
        hidden_dim = self.object_token_dim * 2
        
        # weight standardization
        assert configs["norm_type"] == "group", "Weight standardization is only supported with group normalization."
        conv1d = ws.Conv1d
        
        # normalization
        if configs["norm_type"] == "group":
            self.short_motion_encoder = nn.Sequential(
                conv1d(in_channels=self.object_token_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=configs["n_groups"], num_channels=hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=configs["dropout_p"]),
                conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=configs["n_groups"], num_channels=hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=configs["dropout_p"]),
                conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=configs["n_groups"], num_channels=hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=configs["dropout_p"]),
                conv1d(in_channels=hidden_dim, out_channels=self.lang_token_dim, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups=configs["n_groups"], num_channels=self.lang_token_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=configs["dropout_p"]),
                conv1d(in_channels=self.lang_token_dim, out_channels=self.lang_token_dim, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups=configs["n_groups"], num_channels=self.lang_token_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=configs["dropout_p"]),
                conv1d(in_channels=self.lang_token_dim, out_channels=self.lang_token_dim, kernel_size=1, stride=1, padding=0),
            )
        else:
            raise ValueError(f"Invalid norm type: {configs['norm_type']}")

        # object-language alignment layers
        self.object_lang_align_layers = nn.ModuleList([
            ObjectLanguageAlignmentLayer(configs) for _ in range(self.n_layers)
        ])
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            torch.randn((1, self.lang_token_dim // 2)).float(),
        )

        # negative motion token
        self.negative_token = nn.Embedding(self.n_negative, self.lang_token_dim)

    def get_temporal_positional_encoding(self, x):
        """
        Computes the temporal positional encoding for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n_objects, n_frames, embed_dim].

        Returns:
            torch.Tensor: Temporal positional encoding tensor of shape [batch_size, n_objects, n_frames, embed_dim].
        """
        b, n_obj, t, _ = x.shape
        temporal_pe = torch.arange(t).float().reshape(1, 1, -1, 1).repeat(b, n_obj, 1, 1).to(x.device)
        temporal_pe = temporal_pe / self.max_temporal_length
        temporal_pe = temporal_pe @ self.positional_encoding_gaussian_matrix
        temporal_pe = 2 * np.pi * temporal_pe
        temporal_pe = torch.cat([torch.sin(temporal_pe), torch.cos(temporal_pe)], dim=-1)
        return temporal_pe
    
    def forward(
        self,
        object_tokens,
        lang_tokens,
    ):
        # short-term motion encoding
        b, n_obj, t, d = object_tokens.shape
        object_tokens = object_tokens.permute(0, 1, 3, 2).reshape(b * n_obj, d, t)
        object_tokens = self.short_motion_encoder(object_tokens)
        _, d, t = object_tokens.shape
        object_tokens = object_tokens.reshape(b, n_obj, d, t).permute(0, 1, 3, 2)
        
        # positional encoding
        object_tokens_pe = self.get_temporal_positional_encoding(object_tokens)
        
        # use negative token
        negative_object_tokens = self.negative_token.weight.clone().unsqueeze(0).repeat(b, 1, 1)
        lang_tokens = torch.cat([lang_tokens, negative_object_tokens], dim=1)
        
        # object-language alignment layers
        for layer_idx, layer in enumerate(self.object_lang_align_layers):
            object_tokens, lang_tokens = layer(object_tokens, object_tokens_pe, lang_tokens)
        score_logits = torch.einsum("bntd,bwd->bntw", object_tokens, lang_tokens)
        score_logits = torch.mean(score_logits, dim=-1)
        
        score_token_weight = F.softmax(score_logits, dim=-1)
        
        score_tokens = torch.sum(object_tokens * score_token_weight.unsqueeze(-1), dim=2)
        
        score_map = torch.einsum("bnd,bwd->bnw", score_tokens, lang_tokens)
        score_map = torch.mean(score_map, dim=-1)
        
        return score_map, score_tokens
    
    def get_grad_norm_dict(self):

        grad_norm_dict = {
            "total_grad_norm": 0.0,
            "short_motion_encoder": 0.0,
        }
        grad_norm_dict.update({
            f"scmola_layer_{layer_idx}": 0.0 for layer_idx in range(self.n_layers)
        })
        
        for param in self.short_motion_encoder.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item() ** 2
                grad_norm_dict["short_motion_encoder"] += grad_norm
                grad_norm_dict["total_grad_norm"] += grad_norm
        grad_norm_dict["short_motion_encoder"] = grad_norm_dict["short_motion_encoder"] ** 0.5
        
        for layer_idx, layer in enumerate(self.object_lang_align_layers):
            for param in layer.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm(2).item() ** 2
                    grad_norm_dict[f"scmola_layer_{layer_idx}"] += grad_norm
                    grad_norm_dict["total_grad_norm"] += grad_norm
            grad_norm_dict[f"scmola_layer_{layer_idx}"] = grad_norm_dict[f"scmola_layer_{layer_idx}"] ** 0.5
        
        grad_norm_dict["negative_token"] = 0.0
        for param in self.negative_token.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item() ** 2
                grad_norm_dict["negative_token"] += grad_norm
                grad_norm_dict["total_grad_norm"] += grad_norm
        
        grad_norm_dict["negative_token"] = grad_norm_dict["negative_token"] ** 0.5
        grad_norm_dict["total_grad_norm"] = grad_norm_dict["total_grad_norm"] ** 0.5
        
        return grad_norm_dict