# Image Captioning
This project uses a combined CNN and RNN network to create an image captioning system. This project will use the COCO Dataset and API to train the model with. The COCO API has been provided in the root directory of this repository as `pycocotools`.

## Installation
In addition to the python packages installation instructions that are found at the root of this project repository, the COCO 2014 dataset needs to be downloaded.

1. The COCO 2014 image dataset downloads are facilitated using Google Cloud Platform's (GCP) `gsutil`. This is downloaded by using the command below. Follow the instructions for installation. **You may need to restart your terminal for `gsutil` to work.** 
    ```sh
    curl https://sdk.cloud.google.com | bash
    ```
2. Create data folder where images will be stored in:
    ```sh
    cd image_caption
    mkdir data
    cd data
    ```

3. Download the COCO 2014 training and validation images into the `data` directory (warning: this will download roughly 25GB of data)
    ```sh
    mkdir train2014
    mkdir val2014
    gsutil -m rsync gs://images.cocodataset.org/train2014 train2014
    gsutil -m rsync gs://images.cocodataset.org/val2014 val2014
    ```

2. Unzip the downloaded files
```sh
curl -O http://images.cocodataset.org/zips/train2014.zip
curl -O http://images.cocodataset.org/zips/val2014.zip
curl -O http://images.cocodataset.org/zips/test2014.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```

3. And that's it!
