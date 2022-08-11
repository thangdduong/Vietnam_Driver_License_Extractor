# Vietnam Driver's License Information Extractor System

## Description
This repo is mainly about building a Vietnam driver's license information extractor system. Only with a raw image of the driver's license, the program will recognize the desired information on the card (available on both front and back of the card) and print them on to the screen.

## Installation

### Install required packages
I highly recommend you using Anaconda (or Miniconda) for creating a virtual enviroment, this ensure the program can run perfectly. You can download Anaconda in the following instruction: [download](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html).

After Anaconda is downloaded, create a virtual enviroment and install required libraries:

```bash
$ conda create -n virtual_env -y python=3.8
$ conda activate virtual_env
$ pip install -r requirements.txt
```

### Download file weight
You should download all the files below and put them into `weights` folder:
- Four corners detection: [link](https://drive.google.com/file/d/1ojKv7eBSV9LVZV9lBLhYTe99ZHo-j7U7/view?usp=sharing)
- Text detection: [link](https://drive.google.com/file/d/1QAoSK5GyHaOvzJ8DMClWsM6GPTfoJsa-/view?usp=sharing)
- Text recognition: [link](https://drive.google.com/file/d/10_mdnVbCLHlmR06za32AOCBJ_-PoKIRT/view?usp=sharing)

## Usage
To extract information from a given image, simply run `main.py` with image path:
```
python main.py -i /path/to/your/image
```

