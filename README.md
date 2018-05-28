## Experiments for Broadscale Domain Adaptation Using Adaptive Sampling and Active Learning
----------
Authors:

- [Dr. Ricardo Vilalta](http://www2.cs.uh.edu/~vilalta/)
- [Dainis Boumber](https://www.researchgate.net/profile/Dainis_Boumber)
- [Mikhail M.Meskhi](michaelmm.com)

Dr. Vilalta introduced a complexity metric for domain adaptation tasks. By measuring class entropy for *k*-NN we determine the adaptive sampling technique on the target dataset which we append to the source dataset. Thus, training on a newly formed dataset.

#### Usage

Convert your `.csv` dataset file using the `libsvm` script to support libsvm sparse data. 

` python3 libsvm/libsvm.py data/mnist_full.csv data/mnist_full.txt `

To run a Multilayered Perceptron with Active Learning run the following. For specific sampling strategies and other model configurations, check the model file. 

` python3 models/mlp_active_learning.py data/mnist_full.txt 0.3 1500 500`

The models have Uncertainty sampling strategy set as default. Refer to `libact` library for more details. 