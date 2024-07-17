# COLT
The implementation for CIKM 2024: Towards Completeness-Oriented Tool Retrieval for Large Language Models.

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

If you find our code or work useful for your research, please cite our work.

```
@inproceedings{qu2024colt,
  title={COLT: Towards Completeness-Oriented Tool Retrieval for Large Language Models},
  author={Qu, Changle and Dai, Sunhao and Wei, Xiaochi and Cai, Hengyi and Wang, Shuaiqiang and Yin, Dawei and Xu, Jun and Wen, Ji-Rong},
  booktitle={Proceedings of the 33nd ACM International Conference on Information and Knowledge Management},
  year={2024}
}
```
