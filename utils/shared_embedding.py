import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from dataset import TextDataset, MusicDataset
from torch.utils.data import DataLoader, default_collate
from config import global_model
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import os

device = 'cpu'

def get_embeddings(model, loader, modality='text'):
    
    """
    提取给定 loader 中样本经过 shared transformer block 后的 embedding.
    返回 embedding (N x hidden_dim) 和对应的 label list.
    """
    embeddings, labels = [], []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            # 根据不同模态，构造输入格式
            if modality == 'text':
                input_ids, attention_mask, label = batch[0].to(device), batch[1].to(device), batch[2]
                x = {'input_ids': input_ids, 'attention_mask': attention_mask}
            else:
                # 对于 music 数据，假设返回 (oct_pair, label)
                x, label = batch[0].to(device), batch[1]
            # 假设模型中 shared transformer block 的调用为 model.transformer_block(x)
            shared_out = model(x, type=modality, return_trsfm_embedding=True)  # 输出 shape: [batch_size, hidden_dim]
            embeddings.append(shared_out.cpu().numpy())
            labels.extend(label.cpu().numpy() if isinstance(label, torch.Tensor) else label)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, np.array(labels)

# 计算同一类别下，不同模态之间的平均距离（欧氏距离）
def compute_mean_distance(emb1, emb2):
    # emb1, emb2 shape: [N, dim]
    dists = np.linalg.norm(emb1 - emb2, axis=1)
    return np.mean(dists)

if __name__ == '__main__':
    model = global_model
    # prepare data
    ckpt_dir = 'train_logs/ckpt/epochs-60_batch_size-32_text_max_length-512_music_max_length-1024_sample_size-None_warmup_step-10000_decay_step-100000_lr-0.0001_weight_decay-1e-05_1742624957.1600728'
    model.load_weights(os.path.join(ckpt_dir, 'checkpoints_best.pth'))
    
    music_dataset = MusicDataset(
        set_name='train',
        sample_size=100,
        # txt_list_json=os.path.join(ckpt_dir, 'music/test_data.json')
    )

    music_loader = DataLoader(
        music_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda batch: default_collate([b for b in batch if b is not None])
    )

    text_dataset = TextDataset(
        set_name='train',
        sample_size=100,
    )

    text_loader = DataLoader(
        text_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda batch: default_collate([b for b in batch if b is not None])
    )
    result_dir = 'results/embedding_analysis'
    os.makedirs(result_dir, exist_ok=True)
    
    
    # get embedding
    emb_text, labels_text = get_embeddings(model, text_loader, modality='text')
    emb_music, labels_music = get_embeddings(model, music_loader, modality='music')
    print('Text Embedding Shape: ', emb_text.shape, labels_text.shape)
    print('Music Embedding Shape: ', emb_music.shape, labels_music.shape)
    
    # 合并两个模态的数据，并加上一个标记区分模态
    emb_all = np.concatenate([emb_text, emb_music], axis=0)
    modalities = np.array(["text"] * emb_text.shape[0] + ["music"] * emb_music.shape[0])
    
    labels = ['BEFORE', 'AFTER', 'IS_INCLUDED', 'SIMULTANEOUS']
    labels_text = np.array([labels[l] for l in labels_text])
    labels_music = np.array([labels[l] for l in labels_music])
    labels_all = np.concatenate([labels_text, labels_music], axis=0)
    

    # 对 embedding 做 PCA 降维到2维
    pca = PCA(n_components=2)

    # 生成文本和音乐的 PCA 图像
    def plot_pca(embeddings, labels, modalities, title, filename):
        emb_pca = pca.fit_transform(embeddings)
        df = pd.DataFrame({
            'PC1': emb_pca[:, 0],
            'PC2': emb_pca[:, 1],
            'Label': labels,
            'Modality': modalities
        })
        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=df, x='PC1', y='PC2', hue='Label', style='Modality', s=80)
        plt.title(title)
        plt.savefig(os.path.join(result_dir, filename))
        print(f"PCA plot saved to {filename}")
        plt.close()

    # 全部数据的 PCA 图像
    plot_pca(emb_all, labels_all, modalities, "PCA of Shared Embeddings for Text and Music", 'pca.png')

    # 仅文本数据的 PCA 图像
    plot_pca(emb_text, labels_text, ["text"] * emb_text.shape[0], "PCA of Text Embeddings", 'pca_text.png')

    # 仅音乐数据的 PCA 图像
    plot_pca(emb_music, labels_music, ["music"] * emb_music.shape[0], "PCA of Music Embeddings", 'pca_music.png')

    # 每个标签的文本和音乐的 PCA 图像
    for label in np.unique(labels_all):
        emb_text_label = emb_text[labels_text == label]
        emb_music_label = emb_music[labels_music == label]
        emb_label = np.concatenate([emb_text_label, emb_music_label], axis=0)
        modalities_label = np.array(["text"] * emb_text_label.shape[0] + ["music"] * emb_music_label.shape[0])
        labels_label = [label] * emb_label.shape[0]
        plot_pca(emb_label, labels_label, modalities_label, f"PCA of Embeddings for Label {label}", f'pca_{label}.png')




    unique_labels = np.unique(labels_all)
    mean_dists = []
    for label in unique_labels:
        # 分别取出 text 与 music 中该 label 的 embedding
        emb_text_label = emb_text[labels_text == label]
        emb_music_label = emb_music[labels_music == label]
        # 为了简单起见，这里取两者数量最小的部分，逐一配对计算距离
        n = min(emb_text_label.shape[0], emb_music_label.shape[0])
        if n > 0:
            dist = compute_mean_distance(emb_text_label[:n], emb_music_label[:n])
            mean_dists.append(dist)
        else:
            mean_dists.append(np.nan)

    plt.figure(figsize=(8, 4))
    plt.plot(unique_labels, mean_dists, marker='o')
    plt.xlabel("Temporal Relation Label")
    plt.ylabel("Mean Euclidean Distance\n(text vs music)")
    plt.title("Mean Distance of Shared Embeddings Across Modalities per Label")
    
    plt.savefig(os.path.join(result_dir, 'mean_distance.png'))