# Image Captioning
This project uses a combined CNN and RNN network to create an image captioning system. This project will use the COCO Dataset and API to train the model with. 

## Installation
In addition to the python packages installation instructions that are found at the root of this project repository, the COCO dataset and API need to be downloaded: 

1. Clone COCO API repository
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
