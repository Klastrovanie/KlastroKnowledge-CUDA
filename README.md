# KlastroKnowledge-CUDA
A CUDA extension for Mahalanobis distance-based top-K matching, outperforming cosine similarity on complex multi-attribute queries, RAG retrieval pipelines, and GenAI applications

[![Available on AWS Marketplace](https://img.shields.io/badge/AWS%20Marketplace-Available-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/marketplace/pp/prodview-xbm7jlntjlzry)

First Public Release: 2026-03-16  
Last Updated: 2026-10-21

## Motivation

This library proposes Mahalanobis distance as a principled alternative to cosine similarity for embedding-based matching. Interestingly, fields such as medicine, economics, and social statistics abandoned cosine similarity a long time ago — yet engineering disciplines have continued to rely on it as a default. This peculiar habit in engineering is precisely what motivated us to formalize this approach into a patent and, after validating strong performance in GenAI applications, to release it as open source.  

Mahalanobis distance is named after the Indian statistician Prasanta Chandra Mahalanobis.   
For more details, see Wikipedia (https://en.wikipedia.org/wiki/Prasanta_Chandra_Mahalanobis)

Also, regarding the Mahalanobis distance, there is a highly recommended lecture on this topic is also available on YouTube (https://youtu.be/rBv39pK1iEs?si=JhVeCs2nUTK-8Gdg&t=1346)  by professor Gary King of Harvard University.

## Quick Start

### 1. Conda Environment
```bash
cd anaconda  
conda env create -f environment.yml  
conda activate <env_name>
``` 

### 2. Build from Source
```bash
bash make_package.sh cuda12   # or cuda11, cuda13
```

### CUDA Compatibility

KlastroKnowledge-CUDA has been tested up to CUDA 13.

The package is distributed as source-build software through GitHub. Users are expected to build the CUDA extension in their own environment using a compatible NVIDIA driver, CUDA Toolkit, PyTorch version, and GPU architecture.

Compatibility with future CUDA Toolkit versions, including CUDA 14 or later, may require updates to build flags, PyTorch extension settings, or supported GPU architecture targets.


### 3. Install 
```bash
# After build, go to either of the generated folders, or cd klastroknowledge_cuda12_release  
cd deploy_exercise   
pip install .
```

### Benchmarks

Navigate to the benchmarks folder and run the provided Jupyter notebooks step by step. 

#### CLIP Embeddings (COCO val2017)
- Margin(1-2): **18x larger** than cosine similarity
- Softmax Entropy: **3x lower** than cosine similarity
- Mahalanobis suppresses off-target images that cosine retains near the top,
  because cosine score gaps are too small to discriminate effectively.
   
#### Tabular Data (UCI Adult Dataset, k=25)
- CKA improved from 0.910373 → 0.928622 (**+0.018249 absolute, ~2.00% relative**)
- Mean covariance trace reduced from 59.435885 → 56.828646 (**4.39% average reduction**)
- Consistent advantage observed across k=15 and k=25

### Context Engineering Demo

Through context engineering, KlastroKnowledge was used to identify the optimal match without retraining the AI.

- [Demo #1](https://youtu.be/ALOvoPxTbYA)
- [Demo #2](https://youtu.be/mN0v5Yd8kz0)
- [Demo #3](https://youtu.be/9BXmCXiOXO4)


## License

AGPL v3 — free for research and non-commercial use.  
Commercial use requires a separate agreement.

This code is released to encourage collaboration across AI systems — not competition.  
The goal is shared solutions, not shared resources.

For commercial licensing: leave a message on [Discussions](../../discussions)

## Patent Information
Korean Patent No: 10-2937626  
DOI: https://doi.org/10.8080/1020250102273 

(registered March 6, 2026), titled *"Method and Device for Determining the Reliability of AI Model Outputs."*

**Core claims covered by this patent:**

- Mapping AI model outputs to embedding space to obtain embedding vectors
- Computing a covariance matrix from a reference vector pool
- Calculating **Mahalanobis distance** between reference vectors and embedding vectors based on the covariance matrix
- Selecting top-k reference vectors sorted by Mahalanobis distance
- Assigning ranks to selected k vectors in ascending order
- Computing **softmax entropy** and **probability margin** (rank-1 vs rank-2) from the resulting probability distribution
- Determining output reliability based on entropy and probability margin
- Applicable to text, image, and multimodal domains (CLIP-based embeddings supported)

Any commercial implementation of the above pipeline without a license agreement constitutes patent infringement.

## Pricing

### Open Source — Free
AGPL v3. Research, education, and non-commercial use.
Network use triggers source disclosure obligation under AGPL v3.

### Commercial License

The commercial edition of KlastroKnowledge is available on **AWS Marketplace**:

**[KlastroKnowledge on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-xbm7jlntjlzry)**

It is a production-ready container (MCP server and batch job) that runs in your own AWS account: exact Mahalanobis matching on the GPU, with inputs and outputs on Amazon S3, plan-based job sizes and email support. Plans, pricing and the End User License Agreement are shown on the listing, and billing goes through your AWS account.

The commercial edition is licensed under its own EULA, not under the AGPL v3, so AGPL obligations do not apply to its use.

**OEM, redistribution, or use of this open-source kernel without AGPL v3 obligations** (for example embedding it in products delivered to your customers), including patent licensing (KR Patent 10-2937626): contact **contact@klastrovanie.com**.

#### How to Inquire
For commercial licensing or private offer, contact us via 
[Discussions](https://github.com/Klastrovanie/KlastroKnowledge-CUDA/discussions).
We typically respond within 2 business days.

## Copyright

Copyright © 2026 Klastrovanie Co., Ltd. All rights reserved.
