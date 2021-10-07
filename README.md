## KPCN - Kernel Point Completion Network
KPCN is a learning-based system for point cloud completion consisting of an autoencoder-structured neural network. The encoder uses a local 3D point convolution operator which takes sphere neighborhoods as input and processes them with weights spatially located by a small set of kernel points. In this way, local spatial relationships of the data are considered and encoded efficiently, contrarily to previous shape completion encoder structures which use a global scope during feature extraction. Aside from the rigid version, the convolution operator used also provides a deformable version, that learns local shifts effectively deforming the convolution kernels to make them fit the point cloud geometry. In addition, a regular subsampling per layer method is adapted which in combination with the radius neighbourhoods provides robustness to noise and varying densities. The decoder of the implemented system is a hybrid structure which combines the advantages of fully-connected layers and folding operators, producing a multi-resolution output.

## Installation
For help on installation, refer to <a href="https://github.com/no-materials/kpcn/blob/master/INSTALL.md">INSTALL.md</a>. Note that KPCN has been tested only on Linux. Windows is currently not supported as the code uses tensorflow custom operations. CUDA & cuDNN are required.

## Datasets
#### ShapenetBenchmark2048
Download from <a href="http://download.cs.stanford.edu/downloads/completion3d/dataset2019.zip">this link</a>.

#### kitti
Download KITTI data in the `kitti` folder on <a href="https://drive.google.com/drive/folders/1M_lJN14Ac1RtPtEQxNlCV9e8pom3U6Pa">Google Drive</a>

## Common commands
For the following common commands, path placeholders are used. These are explained here:
* `<saving_path>`: Log directory of the used model. It contains the model's config file, model checkpoints, visualisation plots and training/validation/test results. Name after the timestamp of the creation of the model's instance, i.e. `/kpcn/results/Log_2019-11-13_13-28-41`.
* `<dataset_path>`: Directory which contains unprocessed and processed data (pickle files) of a dataset. In the case of ShapeNetBenchmark2048 it should also contain three `.list` files which enlist the models used for each training/validation/test split.

Replace the path placeholders in the commands below with your relevant ones.
#### Train
```shell
python train_ShapeNetBenchmark2048.py --saving_path <saving_path> --dataset_path <dataset_path> --snap -1  # use snap = -1 to choose last model snapshot
```

#### Test
```shell
python test_model.py --on_val --saving_path <saving_path> --dataset_path <dataset_path> --snap -1
```
* The `on_val` flag denotes the use of the validation split for the purpose of testing. Executing the above command without the `on_val` flag would run the test on the test split. Note that the test split does not contain any ground-truth, and therefore the command ultimately executes a "Similar model retrieval' task.
* The `calc_tsne` flag can also be used in the above command. It enables the code for the calculation and visualisation of the val/test split's latent space T-SNE embedding.
* The `noise` argument can also be used in the above command. It accepts a float which defines the standard deviation of the normal distribution used as additive noise during the data augmentation phase. These can be used for evaluating the robustness of the model.

#### Test kitti registration
Before testing kitti registration with the following command, make sure you have already completed the kitti models using the above script (`test_model.py`), with the `dataset_path` arguement pointing to the `kitti` dataset directory. Upon successful completion, the completed kitti models should reside in the `/<saving_path>/visu/kitti/completions` directory, and so the following command can be run:
```shell
python kitti_registration.py --plot_freq 20 --saving_path <saving_path> --dataset_path <dataset_path>
```
* The argument `plot_freq` specifies the frequency registrations would be plotted
* The script internally uses the ICP algorithm for registration, and so parameters of the ICP algorithm can be adjusted (type `python kitti_registration.py -h` for more options.)

#### Visualise deformations
An interactive mini-application for visualising the rigid and deformable kernel of chosen layers on input partial point clouds. The subsampled point cloud of each chosen layer is also displayed.
```shell
python visualize_deformations.py --saving_path <saving_path> --dataset_path <dataset_path> --snap -1
```
