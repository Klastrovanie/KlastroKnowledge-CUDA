# KlastroKnowledge-CUDA
A CUDA extension for Mahalanobis distance-based top-K matching, outperforming cosine similarity on complex multi-attribute queries, RAG retrieval pipelines, and GenAI applications

First Public Release: 2026-03-16  
Last Updated: 2026-03-17

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

Per-unit annual license. One license required per **business unit 
or legal entity** that uses the software.

| Headcount (per unit) | Standard | OEM |
|----------------------|----------|-----|
| Up to 50 | €3,000 | €30,000 |
| Up to 100 | €8,000 | €60,000 |
| Up to 200 | €20,000 | €120,000 |
| Up to 500 | €50,000 | €250,000 |
| Up to 1,000 | €100,000 | €500,000 |
| 1,000+ | Contact us | Contact us |

**Standard** — Use in your own products within the licensed 
business unit or legal entity. **Redistribution to third parties 
is not permitted.**

**OEM** — Redistribute, white-label, or embed in products 
delivered to your customers.

#### License Terms

- **Per-unit licensing**: One license per using business unit 
  or legal entity.
- **Headcount verification**: Based on publicly disclosed 
  corporate filings.
- **Contract duration**: Pricing tier remains valid for the 
  full contract term. Tier changes apply at the next renewal.
- **Multi-unit customers**: Contact us for volume terms.

#### What's Included

All commercial plans:
- Patent license (KR Patent 10-2937626)
- Removal of AGPL v3 obligations
- Email support

**Standard adds:** Multi-environment deployment, priority support.  
**OEM adds:** Redistribution rights, white-labeling, 
patent indemnification, custom SLA, roadmap input.

#### How to Inquire
For commercial licensing, contact us via 
[Discussions](https://github.com/Klastrovanie/KlastroKnowledge-CUDA/discussions).
We typically respond within 2 business days.

## Copyright

Copyright © 2026 Klastrovanie Co., Ltd. All rights reserved.
