from fairseq.models.roberta import RobertaModel
import torch
import numpy as np

def get_embeddings(checkpoint_file, data_path, txt_file, user_dir='model/musicbert', batch_size=16):
    """
    读取文本文件，使用 MusicBERT 提取 `lm_head` 的输出特征（embedding）。
    
    参数:
      checkpoint_file: 模型 checkpoint 文件路径
      data_path: fairseq 预处理的数据路径
      txt_file: 文本文件路径，每行作为一个样本
      batch_size: 批大小（默认 16）

    返回:
      embeddings_list: list，每个元素是一个样本的 embedding (numpy array)
    """
    
    # **加载 MusicBERT**
    roberta = RobertaModel.from_pretrained(
        '.',
        checkpoint_file=checkpoint_file,
        data_name_or_path=data_path,
        user_dir=user_dir  # 确保 user_dir 目录正确
    )
    # roberta.cuda()
    roberta.eval()
    
    # **读取文本文件**
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()][:2]
    
    # **Tokenize 文本**
    tokenized_texts = [roberta.encode(line) for line in lines]
    
    # **获取最大长度**（用于 padding）
    max_length = 8192

    # **定义 padding 函数**
    def padded(tensor, max_length):
        if tensor.size(0) > max_length:  # 超长就截断
            return tensor[:max_length]
        pad_length = max_length - tensor.size(0)
        return torch.cat([tensor, torch.full((pad_length,), roberta.task.source_dictionary.pad(), dtype=torch.long)])
    
    # **对所有 tokenized 文本进行 padding**
    padded_tokens = torch.stack([padded(t, max_length) for t in tokenized_texts])
    # padded_tokens = torch.stack([padded(t, max_length) for t in tokenized_texts]).cuda()

    # **获取 `lm_head` 的 embedding**
    embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(padded_tokens), batch_size):
            batch = padded_tokens[i: i + batch_size]
            features = roberta.extract_features(batch)  # 获取 Transformer 语义特征
            embeddings_list.extend(features.cpu().numpy())  # 存储为 numpy

    return embeddings_list

# 示例调用：
if __name__ == '__main__':
    checkpoint = 'data/ckpt/checkpoint_last_musicbert_base_w_genre_head.pt'  # 你的模型 checkpoint 文件
    data_path = 'midi'        # fairseq 数据目录
    txt_file = 'midi/midi_test.txt'        # 输入文本文件
    embeddings = get_embeddings(checkpoint, data_path, txt_file)

    print("总样本数:", len(embeddings))
    print("单个样本 embedding 形状:", embeddings[0].shape)  # 应该是 (sequence_length, 768)