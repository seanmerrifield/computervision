# Computer Vision
A collection of my Computer Vision projects using deep neural network architectures built on Pytorch and OpenCV. 

## Installation
`miniconda` is a great python package management system which can be downloaded [here](https://conda.io/miniconda.html). Install this so that you can run `conda` commands from the command line. 


Run the following commands from the command line terminal to download this repository and install the necessary python packages:

1. Clone this repository
```sh
git clone https://github.com/seanmerrifield/computervision
cd computervision
```

2. Create a new `conda` environment
* **Linux or Mac**
```sh
conda create -n computer-vision python=3.6
source activate computer-vision
```

* **Windows**
```sh
conda create -n computer-vision python=3.6
activate computer-vision
```

3. Install PyTorch and torchvision; this will install the latest version of PyTorch.
* **Linux or Mac**
```sh
conda install pytorch torchvision -c pytorch 
```

* **Windows**
```sh
conda install pytorch-cpu -c pytorch
pip install torchvision
```

4. Other dependent packages are installed from the requirements text file (including OpenCV).
```sh
pip install -r requirements.txt
```

5. And that's it!
