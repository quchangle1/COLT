# COLT
The implementation for ACL 2024 Submission: Towards Completeness-Oriented Tool Retrieval for Large Language Models.

## How to run the code
1. Download PLMs from Huggingface and make a folder with the name PLMs
2. Run Semantic Learning:
	> python train_sbert.py
3. Run Collaborative Learning:
	> python train.py -g 0 -m COLT -d tool

## Environment

Our experimental environment is shown below:

```
numpy version: 1.21.6
pandas version: 1.3.5
torch version: 1.13.1
```

## Reference

Our implementations and experiments of Semantic Learning are conducted based on [BEIR](https://github.com/beir-cellar/beir), which is a popular benchmark containing diverse IR tasks. :

```
@inproceedings{
    thakur2021beir,
    title={{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
    author={Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
    booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
    year={2021},
    url={https://openreview.net/forum?id=wCu6T5xFjeJ}
}
```
