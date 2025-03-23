import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from ..dataset import TextDataset, MusicDataset
from torch.utils.data import DataLoader, default_collate
import os
from config import device, model
import seaborn as sns
import pandas as pd


def get_embeddings(loader, modality='text'):
    """
    提取给定 loader 中样本经过 shared transformer block 后的 embedding.
    返回 embedding (N x hidden_dim) 和对应的 label list.
    """
    embeddings, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            # 根据不同模态，构造输入格式
            if modality == 'text':
                input_ids, attention_mask, label = batch[0].to(device), batch[1].to(device), batch[2]
                x = {'input_ids': input_ids, 'attention_mask': attention_mask}
            else:
                # 对于 music 数据，假设返回 (oct_pair, label)
                x, label = batch[0].to(device), batch[1]
            # 假设模型中 shared transformer block 的调用为 model.transformer_block(x)
            shared_out = model.transformer_block(x)  # 输出 shape: [batch_size, hidden_dim]
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
    
    # prepare data
    ckpt_dir = 'train_logs/ckpt/epochs-40_batch_size-32_text_max_length-512_music_max_length-1024_sample_size-8000_warmup_step-1000_decay_step-50000_lr-8e-05_weight_decay-1e-05_1742733002.8590322'

    music_dataset = MusicDataset(
        set_name='test',
        sample_size=100,
        # txt_list_json=os.path.join(ckpt_dir, 'music/test_data.json')
    )

    music_loader = DataLoader(
        music_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False,
        collate_fn=lambda batch: default_collate([b for b in batch if b is not None])
    )

    text_dataset = TextDataset(
        set_name='test',
        sample_size=100,
    )

    text_loader = DataLoader(
        text_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False,
        collate_fn=lambda batch: default_collate([b for b in batch if b is not None])
    )

    # get embedding
    emb_text, labels_text = get_embeddings(text_loader, modality='text')
    emb_music, labels_music = get_embeddings(music_loader, modality='music')

    print(emb_text.shape, labels_text.shape)
    print(emb_music.shape, labels_music.shape)
    # # 合并两个模态的数据，并加上一个标记区分模态
    # emb_all = np.concatenate([emb_text, emb_music], axis=0)
    # modalities = np.array(["text"] * emb_text.shape[0] + ["music"] * emb_music.shape[0])
    # labels_all = np.concatenate([labels_text, labels_music], axis=0)

    # # 对 embedding 做 PCA 降维到2维
    # pca = PCA(n_components=2)
    # emb_pca = pca.fit_transform(emb_all)

    # # 可视化：以不同颜色标记不同类别，并用不同 marker 标识模态

    # df = pd.DataFrame({
    #     'PC1': emb_pca[:, 0],
    #     'PC2': emb_pca[:, 1],
    #     'Label': labels_all,
    #     'Modality': modalities
    # })

    # plt.figure(figsize=(8, 6))
    # sns.scatterplot(data=df, x='PC1', y='PC2', hue='Label', style='Modality', s=80)
    # plt.title("PCA of Shared Embeddings for Text and Music")
    # plt.show()



    # unique_labels = np.unique(labels_all)
    # mean_dists = []
    # for label in unique_labels:
    #     # 分别取出 text 与 music 中该 label 的 embedding
    #     emb_text_label = emb_text[labels_text == label]
    #     emb_music_label = emb_music[labels_music == label]
    #     # 为了简单起见，这里取两者数量最小的部分，逐一配对计算距离
    #     n = min(emb_text_label.shape[0], emb_music_label.shape[0])
    #     if n > 0:
    #         dist = compute_mean_distance(emb_text_label[:n], emb_music_label[:n])
    #         mean_dists.append(dist)
    #     else:
    #         mean_dists.append(np.nan)

    # plt.figure(figsize=(8, 4))
    # plt.plot(unique_labels, mean_dists, marker='o')
    # plt.xlabel("Temporal Relation Label")
    # plt.ylabel("Mean Euclidean Distance\n(text vs music)")
    # plt.title("Mean Distance of Shared Embeddings Across Modalities per Label")
    # plt.show()