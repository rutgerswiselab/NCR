# Neural Collaborative Reasoning
This repository includes the implementation for paper Neural Collaborative Reasoning (NCR):

*Hanxiong Chen, Shaoyun Shi, Yunqi Li, Yongfeng Zhang. 2021. [Neural Collaborative Reasoning](https://arxiv.org/pdf/2005.08129.pdf). 
In Proceedings of the Web Conference 2021 (WWW ’21)*

A step-by-step tutorial of the NCR model based on Jupyter Notebook is availbe at [here on GitHub](https://github.com/bmxitalia/NCRProject). We appreciate the efforts of [@bmxitalia](https://github.com/bmxitalia) for reproducing the model and providing the nicely designed step-by-step tutorial.

## Refernece

For inquiries contact Hanxiong Chen (hanxiong.chen@rutgers.edu) or Yongfeng Zhang (yongfeng.zhang@rutgers.edu)

```
@inproceedings{chen2021neural,
  title={Neural Collaborative Reasoning},
  author={Hanxiong Chen, Shaoyun Shi, Yunqi Li, and Yongfeng Zhang},
  booktitle={Proceedings of the the Web Conference 2021},
  year={2021}
}
```

## Environments

Python 3.6.6

Packages: See in [requirements.txt](https://github.com/rutgerswiselab/NCR/blob/master/requirements.txt)

```
numpy==1.18.1
torch==1.0.1
pandas==0.24.2
scipy==1.3.0
tqdm==4.32.1
scikit_learn==0.23.1
```

## Datasets

- The processed datasets are in  [`./dataset/`](https://github.com/rutgerswiselab/NCR/tree/master/dataset)

- **ML-100k**: The origin dataset can be found [here](https://grouplens.org/datasets/movielens/100k/). 

- **Amazon Datasets**: The origin dataset can be found [here](http://jmcauley.ucsd.edu/data/amazon/). 
    

## Example to run the codes
-   To guarantee the program can execute properly, please keep the directory structure as given in this repository.
-   Some running commands can be found in [`./command/command.py`](https://github.com/rutgerswiselab/NCR/blob/master/command/command.py)
-   For example:

```
# Neural Collaborative Reasong on ML-100k dataset
> cd NCR/src/
> python main.py --rank 1 --model_name NCR --optimizer Adam --lr 0.001 --dataset ml100k01-1-5 --metric ndcg@5,ndcg@10,hit@5,hit@10 --max_his 5 --test_neg_n 100 --l2 1e-4 --r_weight 0.1 --random_seed 2022 --gpu 0
```
