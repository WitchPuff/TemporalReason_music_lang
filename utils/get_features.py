from fairseq.models.roberta import RobertaModel
import torch
import random

def init_musicbert(checkpoint_file, data_path, user_dir='model/musicbert'):
    print('Loading MusicBERT...')
    roberta = RobertaModel.from_pretrained(
        '.',
        checkpoint_file=checkpoint_file,
        data_name_or_path=data_path,
        user_dir=user_dir  # 确保 user_dir 目录正确
    )
    return roberta

def get_music_embeddings(roberta, txt_file, batch_size=2, max_length=8192, random_pair=True):
    """
    读取文本文件，使用 MusicBERT 提取 Transformer (`extract_features`) 的输出特征（embedding）。
    
    参数:
      checkpoint_file: 模型 checkpoint 文件路径
      data_path: fairseq 预处理的数据路径
      txt_file: 文本文件路径，每行作为一个样本
      batch_size: 批大小

    返回:
      embeddings_list: list，每个元素是一个样本的 embedding (numpy array)
    """
    

    roberta.eval()
    
    # **读取文本文件**
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    # **Tokenize 文本**
    tokenized_texts = [roberta.task.source_dictionary.encode_line(line, append_eos=True, add_if_not_exist=False) for line in lines]
    # **定义 padding 函数**
    def padded(tensor, max_length=8192):
        if tensor.size(0) > max_length:  # 超长就截断
            return tensor[:max_length]
        pad_length = max_length - tensor.size(0)
        return torch.cat([tensor, torch.full((pad_length,), roberta.task.source_dictionary.pad(), dtype=torch.long)])
    
    # **对所有 tokenized 文本进行 padding**
    padded_tokens = torch.stack([padded(t, max_length) for t in tokenized_texts]).to(device)

    # **获取 `extract_features` 的 embedding**
    features = roberta.extract_features(padded_tokens)  # 获取 Transformer 语义特征
    if random_pair:
        random.shuffle(features)
    return features.detach().to('cpu')



# 示例调用：
if __name__ == '__main__':
    checkpoint = 'data/ckpt/checkpoint_last_musicbert_base_w_genre_head.pt'  # 你的模型 checkpoint 文件
    data_path = 'midi'        # fairseq 数据目录
    txt_file = 'data/midi_oct/train/during/z4zmgmsULaYsoD1F8VQDHHr6GdiJwAKJclNGlyffmRzDgBT6h4gdebWhF3GOz7Tk0caS52X4JAWArYpY9mdldcGSLIE4k8w2DGuNqX3q2Cw6wFtm9wOMes0KrcFA3nImMtmTg_1.txt'        # 输入文本文件
    
        # **自动检测设备**
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
        print("使用 CUDA 进行计算")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Mac M1/M2
        print("使用 MPS（Mac GPU）进行计算")
    else:
        device = torch.device("cpu")  # CPU
        print("使用 CPU 进行计算")
    roberta = init_musicbert(checkpoint, data_path)
    embeddings = get_music_embeddings(roberta, txt_file, device)

    print("总样本数:", len(embeddings))
    print("单个样本 embedding 形状:", embeddings[-1].shape)  # 应该是 (sequence_length, 768)