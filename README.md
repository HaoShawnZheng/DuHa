# DuHa: a dual-hand action segmentation method for human-robot collaborative assembly

This repository is the official repository made for the paper **DuHa: a dual-hand action segmentation method for human-robot collaborative assembly**. It contains the necessary data and code to replicate DuHa.

## Data preparation
The data is hosted in [Dropbox](https://www.dropbox.com/scl/fo/ura9rs952qploe73kyrk3/h?rlkey=9xl2ur3iojl69x1oay9ko2xuo&dl=0). We created the dataset by selecting a sub-dataset from [HA-ViD](https://iai-hrc.github.io/ha-vid). In this repository, we only provide the features and labels that are necessary to replicate DuHa. More information about the dataset **HA-ViD** can be found at the [website](https://iai-hrc.github.io/ha-vid).
You should download the data and put it in a folder `./data`. 
The structure of `data` should look like:
```
data
├── train_features
├── train_edge_indices
├── train_i3d_features
├── train_lh_labels
├── train_rh_labels
├── test_features
├── test_edge_indices
├── test_i3d_features
├── test_lh_labels
├── test_rh_labels
```

It contains the features (bboxes for all object), edge_indices (edges between objects in the graph we introduced in paper), i3d features (scene features in the paper), lh_labels (action labels for left hand), and rh_labels (action labels for right hand) for both training set and testing set.

## Training and testing DuHa
To simplify the process, we use one script `main.py` to automatically train and test DuHa. We test DuHa after each epoch. In the HA-ViD, the videos have `front`, `side` and `top` views, are denoted as `M0` `S1` and `S2` respectively. To run the script `main.py`, please specify the `view` and `data_root` where you hold the data.
* run `python main.py --view M0 --data_root ./data/`

## Check logs
The log files will be save in `./log` dictionary. It contains dual-hand action segmentation accuracy of each epoch. 

## Citation
If you find our code useful, please cite our paper. 
```
@inproceedings{
  author    = {Hao Zheng and
               Regina Lee and
               Yuqian Lu and 
               Xun Xu},
  title     = {DuHa: a dual-hand action segmentation method for human-robot collaborative assembly},
  journal = {}
}
```

## Contact
If you have any question about DuHa, please contact Hao Zheng via [email](hzhe951@aucklanduni.ac.nz).
