# KlastroKnowledge-CUDA
A CUDA extension for Mahalanobis distance-based top-K matching, outperforming cosine similarity on complex multi-attribute queries, RAG retrieval pipelines, and GenAI applications

First Public Release: 2026-03-16  
Last Updated: 2026-03-16

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


## License

AGPL v3 — free for research and non-commercial use.  
Commercial use requires a separate agreement.

This code is released to encourage collaboration across AI systems — not competition.  
The goal is shared solutions, not shared resources.

For commercial licensing: leave a message on [Discussions](../../discussions)

## Patent Information
Korean Patent No: 10-2937626  
DOI: https://doi.org/10.8080/1020250102273 

## Copyright

Copyright © 2026 Klastrovanie Co., Ltd. All rights reserved.
