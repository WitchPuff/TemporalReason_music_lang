import torch
import torch.nn as nn
from typing import Tuple
from fairseq.models.roberta import RobertaModel
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM



# class SelfAttention(nn.Module):
#     """
#     Intra-modal self-attention module
#     """
#     def __init__(self, hidden_dim=768, num_heads=8):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, 
#                                                 num_heads=num_heads, 
#                                                 batch_first=True)

#     def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
#         # [batch_size, 2, seq_len, hidden_dim]
#         x1 = embeddings[:, 0]
#         x2 = embeddings[:, 1]

#         # x1 <-> x2
#         attn_1, _ = self.self_attn(x1, x2, x2)
#         attn_2, _ = self.self_attn(x2, x1, x1)

#         # Residual
#         x1 = x1 + attn_1
#         x2 = x2 + attn_2
        
#         # [batch_size, 2, seq_len, hidden_dim]
#         x = torch.stack([x1, x2], dim=1)  
#         return x



# class CrossAttentionModule(nn.Module):
#     """
#     Inter-modal cross-attention module
#     """
#     def __init__(self, hidden_dim=768, num_heads=8):
#         super().__init__()
#         self.m2t_attn = nn.MultiheadAttention(embed_dim=hidden_dim, 
#                                                         num_heads=num_heads, 
#                                                         batch_first=True)
#         self.t2m_attn = nn.MultiheadAttention(embed_dim=hidden_dim, 
#                                                         num_heads=num_heads, 
#                                                         batch_first=True)
#         self.layer_norm = nn.LayerNorm(hidden_dim)
#         self.pooling = nn.AdaptiveAvgPool1d(1)

#     def forward(self, xm: torch.Tensor, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             - xm: [batch_size, 2, 8192, hidden_dim]
#             - xt:  [batch_size, 2, seq_len_text,  hidden_dim]

#         Output:
#             - m: [batch_size, hidden_dim]
#             - t:  [batch_size, hidden_dim]
#         """
#         m1 = xm[:, 0]  # [batch_size, 8192, hidden_dim]
#         m2 = xm[:, 1]
#         t1 = xt[:, 0]  # [batch_size, seq_len_text, hidden_dim]
#         t2 = xt[:, 1]

#         m1_attn, _ = self.m2t_attn(m1, t2, t2)
#         m2_attn, _ = self.m2t_attn(m2, t1, t1)

#         t1_attn, _ = self.t2m_attn(t1, m2, m2)
#         t2_attn, _ = self.t2m_attn(t2, m1, m1)

#         m1 = self.layer_norm(m1 + m1_attn)
#         m2 = self.layer_norm(m2 + m2_attn)
#         t1 = self.layer_norm(t1 + t1_attn)
#         t2 = self.layer_norm(t2 + t2_attn)

#         m1 = self.pooling(m1.transpose(1, 2)).squeeze(-1)
#         m2 = self.pooling(m2.transpose(1, 2)).squeeze(-1)
#         t1 = self.pooling(t1.transpose(1, 2)).squeeze(-1)
#         t2 = self.pooling(t2.transpose(1, 2)).squeeze(-1)

#         # 计算最终表示
#         m = (m1 + m2) / 2.0
#         t = (t1 + t2) / 2.0

#         return m, t



# class FusionGate(nn.Module):
#     """
#     Inter-modal fusion gate
#     """
#     def __init__(self, hidden_dim=768):
#         super().__init__()
#         self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, xm: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
#         """
#         xm: [batch_size, hidden_dim]
#         xt:  [batch_size, hidden_dim]
#         """
#         x = torch.cat([xm, xt], dim=-1)
#         r = self.sigmoid(self.gate(x))
#         return r * xm + (1 - r) * xt



class TaskClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=7):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)



class SharedTransformerBlock(nn.Module):
    
    def __init__(self, hidden_dim=768, num_heads=8, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Input: [batch_size, seq_len, hidden_dim]
        Output: [batch_size, hidden_dim]
        """
        for layer in self.layers:
            embeddings = layer(embeddings)
        pooled = self.pooling(embeddings.transpose(1, 2)).squeeze(-1)  # [batch_size, hidden_dim]
        return pooled

class MusicEncoder(nn.Module):

    def __init__(self,
                 checkpoint_file='data/ckpt/checkpoint_last_musicbert_base_w_genre_head.pt',
                 data_path='data/midi_oct',
                 user_dir='model/musicbert',
                 random_pair=True):
        super().__init__()
        self.musicbert = RobertaModel.from_pretrained(
            '.',
            checkpoint_file=checkpoint_file,
            data_name_or_path=data_path,
            user_dir=user_dir
        )
        # freeze the params of MusicBERT
        for param in self.musicbert.parameters():
            param.requires_grad = False
        self.random_pair = random_pair
        
    
        
    def forward(self, oct: torch.Tensor) -> torch.Tensor:

        features = self.musicbert.extract_features(oct)

        return features.detach()


class TextEncoder(nn.Module):
    def __init__(self, 
                 model_name="FacebookAI/roberta-base"):
        super().__init__()
                # Load model directly

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        self.model.eval()
    
        feature = self.model(**x, output_hidden_states=True)
        feature = feature.hidden_states[-1]

        # last_4_layers = torch.stack(hidden_states[-4:])  # (4, batch_size, seq_len, hidden_dim)
        # feature = torch.mean(last_4_layers, dim=0)
        
        return feature.detach()

class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class SharedModel(nn.Module):

    def __init__(self, mf_param: dict = None, tf_param: dict=None,
                    hidden_dim=768, num_heads=8, num_layers=4,
                    text_num_classes=7, music_num_classes=7):
        super().__init__()
        self.music_encoder = MusicEncoder(**mf_param)
        self.text_encoder = TextEncoder(**tf_param)
        self.transformer_block = SharedTransformerBlock(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)
        self.ffn = FeedForwardLayer()
        self.text_classifier = TaskClassifier(input_dim=hidden_dim, num_classes=text_num_classes)
        self.music_classifier = TaskClassifier(input_dim=hidden_dim, num_classes=music_num_classes)

    def save_weights(self, path):
        weights_to_save = {
            "transformer_block": self.transformer_block.state_dict(),
            "ffn": self.ffn.state_dict(),
            "text_classifier": self.text_classifier.state_dict(),
            "music_classifier": self.music_classifier.state_dict()
        }
        torch.save(weights_to_save, path)
        print(f"Weights saved to {path}.")

    def load_weights(self, path, strict=True):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))  # Ensure compatibility
        self.transformer_block.load_state_dict(checkpoint["transformer_block"], strict=strict)
        self.ffn.load_state_dict(checkpoint["ffn"], strict=strict)
        self.text_classifier.load_state_dict(checkpoint["text_classifier"], strict=strict)
        self.music_classifier.load_state_dict(checkpoint["music_classifier"], strict=strict)
        print(f"Weights loaded from {path}.")
    
    def forward(self, x, type='text'):

        if type == 'music':
            x0 = self.music_encoder(x[:, 0, :])
            x1 = self.music_encoder(x[:, 1, :])
            x = torch.cat([x0, x1], dim=1)
        else:
            x = self.text_encoder(x)
            
        x = self.transformer_block(x)
        x = self.ffn(x)

        if type == 'music':
            y = self.music_classifier(x)
        else:
            y = self.text_classifier(x)

        return y
