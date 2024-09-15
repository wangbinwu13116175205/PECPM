# PECMP

Code for *Pattern Expansion and Consolidation on Evolving Graphs for Continual Traffic Prediction**（KDD 2023). PECPM is a continual traffic flow forecasting framework, achieving accurate predictions and high efficiency.

### Requirements

* python = 3.8.5
* pytorch = 1.7.1
* torch-geometric = 1.6.3

```
conda env create -f PECMP.yaml
```
  
### Data

Download raw data from [this](https://drive.google.com/file/d/1P5wowSaNSWBNCK3mQwESp-G2zsutXc5S/view?usp=sharing), unzip the file and put it in the `data` folder

### Usages

*Data Process
```
Please use data_process.ipynb to process the data first.
```

* PECMP
```
python main.py --conf conf/PECMP.json --gpuid 1
```


