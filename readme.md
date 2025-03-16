# Progress

13-way classification. **Allen’s Interval Algebra** 提供了 13 种常见的时间区间关系，例如 *before*、*meets*、*overlaps*、*starts*、*during*、*finishes*、*equals* 以及它们的对应逆关系 (after, met-by, overlapped-by, started-by, contained-by, finished-by)

- [ ] Data
  - [x] Music Octuples
  - [ ] English corpus?
  - [ ] Make dataset
    - [ ] Music
      - [ ] x: two sequences of music embeddings (simply concat, switch orders randomly as data augmentation)
      - [ ] y: time relation label
    - [ ] Speech
      - [ ] x: two sequences of speech embeddings
      - [ ] y: time relation label
- [ ] Model
  - [ ] Get Features
    - [x] Pretrained MusicBERT
    - [ ] Pretrained BERT
  - [ ] Shared Attention Embedding Module
  - [ ] Classifiers
- [ ] Train
- [ ] Analysis



```mermaid
graph TD
	%% Input nodes
    A0[Text Input] -.-> A1[Pretrained BERT]
    B0[Music Input] -.-> B1[Pretrained MusicBERT]
    subgraph Language    Branch
    	A1[Pretrained BERT]
    end
subgraph Music   Branch
	B1[Pretrained MusicBERT]
end

subgraph Shared Attention Embedding Module
    A1[Pretrained BERT] -.->|Language Embeddings| C1[Attention Mechanism]
    B1[Pretrained MusicBERT] -.->|Music Embeddings| C1[Attention Mechanism]
end

subgraph Classifiers
    C1 -.->|Shared Embedding of Language| D1[Language Task Classifier]
    C1 -.->|Shared Embedding of Music| D2[Music Task Classifier]
end
 D1 -.-> O[Relations Label]
 D2 -.-> O[Relations Label]

%% Graph styling
classDef module fill:#f9f,stroke:#333,stroke-width:2px;
classDef embedding fill:#ffc,stroke:#333,stroke-width:1px,font-style:italic;
classDef shared fill:#bbf,stroke:#333,stroke-width:2px,font-weight:bold;

class A1,B1,D1,D2 module;
class C1,C2 shared;
```

