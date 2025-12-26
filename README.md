# AIGer

[AIGer](https://ieeexplore.ieee.org/document/11300286) is a GNN-Based Modeling Approach for AIGs.

## Abstract
The automation of logic circuit design enhances chip
performance, energy efficiency, and reliability, and is widely
applied in the field of Electronic Design Automation (EDA). And-
Inverter Graphs (AIGs) efficiently represent, optimize, and verify
the functional characteristics of digital circuits, enhancing the
efficiency of EDA development. Due to the complex structure
and large scale of nodes in real-world AIGs, accurate modeling
is challenging, leading to existing work lacking the ability
to jointly model functional and structural characteristics, as
well as insufficient dynamic information propagation capability.
To address the aforementioned challenges, we propose AIGer,
with the aim to enhance the expression of AIGs and thereby
improve the efficiency of EDA development. Specifically, AIGer
consists of two components: 1) Node logic feature initialization
embedding component and 2) AIGs feature learning network
component. The node logic feature initialization embedding
component projects logic nodes, such as AND and NOT, into
independent semantic spaces, to enable effective node embedding
for subsequent processing. Building upon this, the AIGs feature
learning network component employs a heterogeneous graph
convolutional network, designing dynamic relationship weight
matrices and differentiated information aggregation methods to
better represent the original structure and information of AIGs.
The combination of these two components enhances AIGer’s
ability to jointly model functional and structural characteristics
and improves its message passing capability, thereby strengthen-
ing its expressive power for AIGs. Experimental results indicate
that AIGer outperforms the current best models in the Signal
Probability Prediction (SPP) task, improving MAE and MSE by
18.95% and 44.44%, respectively. In the Truth Table Distance
Prediction (TTDP) task, AIGer achieves improvements of 33.57%
and 14.79% in MAE and MSE, respectively, compared to the
best-performing models.

## Environmental Configuration

**Experimental Environment Installation Package Version Requirements:**

`torch==2.2.1+cu118 `

`torch-sparse==0.6.18`

`torch_scatter==2.1.2`

`numpy==1.26.4`

**Experimental Equipment：**

Ubuntu 22.04.5 LTS

Nvidia A6000

## Experimental Data and parameters

**This is the data required for the experiment:**

https://huggingface.co/datasets/Ichont/AIGer_Dataset

**Explanation:**

- `--task 'prob'`: Select signal probability prediction(SPP) task.
- `--model 'AIGer'`: Use AIGer model.
- `--batch_size 256`: Set the batch size required for training..
- `--split_file 0.05-0.05-0.9`: Indicates the data split proportions for training, validation, and testing.
- `--layer_num 9`: Set the number of layers of the AIGer network to 9..



## Run AIGer

Proper configuration of the bash environment is required.

`bash train.sh`

## Cite
```
@ARTICLE{11300286,
  author={Sun, Weihao and Guo, Shikai and Wang, Siwen and Ma, Qian and Li, Hui and Wang, Ning and Weng, Yongpeng},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
  title={Modeling Relational Logic Circuits for And-Inverter Graph Convolutional Network}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Integrated circuit modeling;Logic;Logic gates;Computational modeling;Representation learning;Predictive models;Design automation;Logic circuits;Optimization;Message passing;Electronic Design Automation;And-Inverter Graphs;Logic Circuit Design},
  doi={10.1109/TCAD.2025.3644279}}
```
