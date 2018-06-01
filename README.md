## Experiments for Broadscale Domain Adaptation Using Adaptive Sampling and Active Learning

## Authors:

- [Dr. Ricardo Vilalta](http://www2.cs.uh.edu/~vilalta/)
- [Dainis Boumber](https://www.researchgate.net/profile/Dainis_Boumber)
- [Mikhail M.Meskhi](michaelmm.com)

Dr. Vilalta introduced a complexity metric for domain adaptation tasks. By measuring class entropy for *k*-NN we determine the adaptive sampling technique on the target dataset which we append to the source dataset. Thus, training on a newly formed dataset.

## Installation

To install all the dependencies run the following command: 

`pip install -r requirements.txt`


## Usage

Convert your `.csv` dataset file using the `libsvm` script to support libsvm sparse data: 

` python3 libsvm/libsvm.py data/mnist_100.csv data/mnist_100.txt `

We have combied all the models into one script `bsda.py`. Just run the script with the following flags:

- sys.argv[1] = dataset path
- sys.argv[2] = validation set size
- sys.argv[3] = prelabled instances
- sys.argv[4] = querying budget

` python3 bsda.py data/mnist_100.txt 0.5 15 10`

The results will be displayed in the termianl along with the standard deviation of accuracy over total querying budget. Currently the tests run over MLP and SVM models. The models have Uncertainty sampling strategy set as default. Refer to `libact` library for more details. 

## Citing

These experiments are based on the Active Learning Library `libact` from: 

```
@techreport{YY2017,
  author = {Yao-Yuan Yang and Shao-Chuan Lee and Yu-An Chung and Tung-En Wu and Si-An Chen and Hsuan-Tien Lin},
  title = {libact: Pool-based Active Learning in Python},
  institution = {National Taiwan University},
  url = {https://github.com/ntucllab/libact},
  note = {available as arXiv preprint \url{https://arxiv.org/abs/1710.00379}},
  month = oct,
  year = 2017
}
```