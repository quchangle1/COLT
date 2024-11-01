# COLT
The implementation for CIKM 2024: Towards Completeness-Oriented Tool Retrieval for Large Language Models.

## News
- **[2024/8/17]** The processed **ToolBench** dataset and **checkpoints** of first-stage semantic learning are released! Please find the processed datasets and checkpoints on [HuggingFace](https://huggingface.co/Tool-COLT).
- **[2024/7/17]** Our [**code**](https://github.com/quchangle1/COLT/tree/main) and [**ToolLens**](https://github.com/quchangle1/COLT/tree/main/datasets/ToolLens) dataset is released.
- **[2024/7/16]** COLT is accepted by [**CIKM 2024**](https://cikm2024.org/).
- **[2024/5/25]** Our [**paper**](https://arxiv.org/abs/2405.16089) is released.

## Quick Start
1. Download PLMs from Huggingface and make a folder with the name PLMs
- **ANCE**: The PLM and description is avaliable [here](https://huggingface.co/sentence-transformers/msmarco-roberta-base-ance-firstp).
- **TAS-B**: The PLM and description is avaliable [here](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b).
- **co-Condensor**: The PLM and description is avaliable [here](https://huggingface.co/sentence-transformers/msmarco-bert-co-condensor).
- **Contriever**: The PLM and description is avaliable [here](https://huggingface.co/nthakur/contriever-base-msmarco).
2. Run Semantic Learning:
	> python train_sbert.py
3. Run Collaborative Learning:
	> python train.py -g 0 -m COLT -d ToolLens

You can specify the gpu id, the used dataset by cmd line arguments.

## Environment

Our experimental environment is shown below:

```
numpy version: 1.21.6
pandas version: 1.3.5
torch version: 1.13.1
```

### Semantic Learning 

For the first phase of semantic learning, you need to install the **BEIR**. BEIR is a widely used information retrieval framework. You can find the BEIR code repository at this [link](https://github.com/beir-cellar/beir).

## Citation

If you find our code or work useful for your research, please cite our work.

```
@inproceedings{qu2024towards,
  title={Towards Completeness-Oriented Tool Retrieval for Large Language Models},
  author={Qu, Changle and Dai, Sunhao and Wei, Xiaochi and Cai, Hengyi and Wang, Shuaiqiang and Yin, Dawei and Xu, Jun and Wen, Ji-Rong},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages={1930--1940},
  year={2024}
}
```
