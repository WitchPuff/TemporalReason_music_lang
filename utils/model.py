import torch
import torch.nn as nn
from fairseq.models.roberta import RobertaModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os


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
                 model_name="data/ckpt/roberta-base"):
        super().__init__()

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
                    text_num_classes=4, music_num_classes=4):
        super().__init__()
        self.music_encoder = MusicEncoder(**mf_param if mf_param else {})
        self.text_encoder = TextEncoder(**tf_param if tf_param else {})
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
        os.makedirs(os.path.dirname(path), exist_ok=True)
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

if __name__ == '__main__':
    model = SharedModel()
    print(model)
