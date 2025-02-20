# CLARITY CPC1 - INSTALLATION + USAGE

The following instructions explain how to install and use the CPC1 software.

The software has been tested on Mac OS and Linux systems, but if you experience any problems please [contact us](http://claritychallenge.org/sign-up-to-the-challenges) or post a query to the challenge [Google group](https://groups.google.com/g/clarity-challenge?pli=1).

The code supplied will produce baseline predictions using the MSBG hearing loss model and MBSTOI intelligibility metric that were released as part of the Clarity Enhancement Challenge.

## 0. Prerequisites

You will need,

- Python (3.7 or more recent)
- ~20 GB of disk space for the challenge data

Visit the `install/` directory.

## 1. Installation

Visit the `install/` directory and run the `install.sh` script.

```bash
cd install
./install.sh
```

This script will:

- Set up a Python virtual environment and install required Python packages.
- Add a top-level data directory data/clarity_data that will be pointed to some inbuilt test data. You can redirect this link when you have installed the main challenge data.

## 2. Install the challenge data

The challenge data is a 17 GB download containing a large sample of hearing aid processed scenes that have been listened to by our panel of hearing impaired listeners. These data are being provided for you to train your prediction model. Later we will be releasing a further set of evaluation data.

First, visit the [download site](https://mab.to/R6H84YNf74p5U) and download the following data pack

- `clarity_CPC1_data.v1_1.tgz`  [17 GB]
- `clarity_CPC1_data.test.v1.tgz`  [10 GB]

To download: click on the file to select it and then click on the download icon in the top-right of the interface.

Unpack the data into a root directory of your choosing using the `unpack.sh` script provided

```bash
cd install
./unpack.sh <DOWNLOAD_DIR>/clarity_CPC1_data.v1_1.tgz <TARGET_DIR>
```

where `DOWNLOAD_DIR` is the location of the download and `<TARGET_DIR>` is the desired location of the unpacked data. The script will produce a directory `<TARGET_DIR>/clarity_CPC1_data` containing the data.

IMPORTANT: Do not untar the package by hand. In addition to untarring, the `unpack.sh` script adds a couple of important symbolic links from the installation directory to the downloaded data.

> ***Note***, if you have previously downloaded the earlier `clarity_CPC1_data.v1.tgz`, you will need to update the metadata. The updated metadata is included in this repository and just needs to be copied into place. See `install/patched/README.md` for details.

## 3. Check the integrity of the data

A simple script is available that will check that the data is correctly installed.
It can be run using,

```bash
python3 scripts/check_data.py data/clarity_data/
```

## 4. Compile c code for the BEZ2018 model

Open MATLAB and go to the following directory `projects/BEZ2018_CUDA`

Execute the file `mexANmodel.m`

Do not forget to include a path or a command to excecute matlab from the command line:

- open `scripts/paths.sh` 
- modify this line: `export MATLAB_BIN="<MATLAB_PATH OR COMMAND>"`

## 5. Compile cu code for the BEZ2018 model (ONLY NVIDIA GPUs)

In total, there are 3 kernels that must be compiled with nvidia toolkit to take advantage of the GPU.

It is important to make sure that the cuda toolkit is well installed. To check it type `nvcc --version` in the terminal. It should return the version of the cuda compiler.

More information can be found [here](https://developer.nvidia.com/cuda-toolkit)

```bash
cd projects
nvcc -ptx BEZ2018_GPU/model_IHC_BEZ2018.cu
nvcc -ptx BEZ2018_GPU/model_Synapse_BEZ2018.cu
nvcc -ptx SAMII/mutual_info.cu
```

## 6. Setup matlab API in python

This step is required in case you choose to use the python scripts, otherwise you could use only bash files.

Steps to install the API is simple and is described [here](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

```bash
cd $MATLAB_ROOT$/extern/engine/python
python setup.py install
```

Make sure to perform this step whithin the CPC1 python environment!!

## 7. Generate predicted intelligibility scores

A baseline intelligibility model is provided. The model is based on a combination of the MSBG hearing loss model and the MBSTOI intelligibility metric. The same model was previously used in the CEC1 enhancement challenge in the objective evaluation of the hearing aid algorithms that were submitted.

For details on how to run the baseline see [scripts/README.md](scripts/README.md).
