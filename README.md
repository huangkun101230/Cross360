<p align="center">

  <h1 align="center">Cross360: 360° Monocular Depth Estimation via Cross Projections Across Scales</h1>
  <p align="center">
    <a href="https://github.com/huangkun101230">Kun Huang</a>,
    <a href="https://people.wgtn.ac.nz/fanglue.zhang?_ga=2.161972092.1710887990.1730665987-888529436.1730407824">Fang-Lue Zhang*</a>,
    <a href="https://people.wgtn.ac.nz/neil.dodgson?_ga=2.172996195.1710887990.1730665987-888529436.1730407824">Neil A. Dodgson</a>,
  </p>
    <p align="center">
    *Corresponding authors

  </p>
  <h3 align="center"><a href="">Paper</a>
  <div align="center"></div>
</p>


## Installation
Provide installation instructions for your project. Include any dependencies and commands needed to set up the project.

```shell
# Clone the repository
git clone https://github.com/huangkun101230/Cross360.git
cd Cross360

# Install dependencies
conda env create -f cross360_conda_env.yml
conda activate cross360
```


## Running
Please [download our pretrained models](https://drive.google.com/drive/folders/1WyuX7Zn649tgV6EVgGo-QhJ4A8UTvN1O?usp=sharing), and save these models to "saved_models/models".
To test on provided data in "./input_data"
```shell
python evaluate.py
```
The results will be saved at "./results/saved_models/"

Please note, we only provided 3 Matterport3D, 3 Stanford2D3D, and 3 SunCG examples in our folder for limited space.

For training our model, please modify the path in our dataset:
For example, in datasets/dataset3D60.py, function gather_filepaths, change local="./input_data/" with your downloaded path

and run
```shell
python train.py
```

## Dataset
We mainly evaluate our method on [M3D](https://niessner.github.io/Matterport/), S2D3D, [3D60 dataset](https://vcl3d.github.io/3D60/) and [Structured3D dataset](https://structured3d-dataset.org/).


## Citation
If you find this repository useful in your project, please cite the following work. :)
```
coming soon.
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.

You are free to:

✅ Share — copy and redistribute the material in any medium or format

✅ Adapt — remix, transform, and build upon the material

Under the following terms:

🔗 Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

🚫 NonCommercial — You may not use the material for commercial purposes.

Note: For any commercial use or licensing inquiries, please contact the project maintainer.
