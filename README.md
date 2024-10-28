# PECMP

Code for **Pattern Expansion and Consolidation on Evolving Graphs for Continual Traffic Prediction**（KDD 2023). PECPM is a continual traffic flow forecasting framework, achieving accurate predictions and high efficiency. We propose an efficient and effective continual learning framework to achieve continuous traffic flow prediction without the access to historical graph data, namely Pattern Expansion and Consolidation based on Pattern Matching (PECPM). Specifically, we first design a bank module based on pattern matching to store representative patterns of the road network. With the expansion of the road network, the model configured with such a bank module can achieve continuous traffic prediction by effectively managing patterns stored in the bank. The core idea is to continuously update new patterns while consolidating learned ones.

### Requirements

* python = 3.8.5
* pytorch = 1.7.1
* torch-geometric = 1.6.3

### Data

Download raw data from [this](https://drive.google.com/file/d/1P5wowSaNSWBNCK3mQwESp-G2zsutXc5S/view?usp=sharing), unzip the file and put it in the `data` folder

### Usages

* Data Process
```
Download attention data from [this](https://pan.baidu.com/s/1JRuYBT0RsRaF11-QI8soKg), Code is mm7w, and unzip the file and put it in the `data` folder


* PECMP
```
python main.py --conf conf/PECMP.json --gpuid 1
```

### Expand and consolidate performance evaluation
```
Select conflicting and stable nodes to evaluate performance.
```


