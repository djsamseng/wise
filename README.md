# CTorch

## Run C++ Torch with CUDA GPU

```bash
cd app
mkdir build
cd build
# If needed: create conda environment for torch dependencies
conda create --name torch python=3.8
# Activate conda environment
conda activate torch
# If needed: Install necessary libraries
conda install -c anaconda cudatoolkit
conda install -c conda-forge cudatoolkit-dev
conda install -c anaconda cudnn
# If needed: Download libtorch with correct CUDA from pytorch.org and unzip. The path to which is used below
# Generate build files
cmake -DCMAKE_PREFIX_PATH=/home/samuel/dev/ctorch/libtorch ..
# Build
cmake --build . --config Release
# Run
./ctorch-app
```

## Visual Studio Code Conda Environemnt
- Ctrl+Shift+P
- Python: Select Intrepreter
- Workspace
- Python 3.8.12 64-bit ('torch': conda)

## boost_app
```bash
cd boost_app
mkdir build
cd build

cmake ..
cmake --build .
./boost-app
```

```bash
cd ../
conda activate copye
python reader.py # Press enter to send BBBBB back
```

## load_library
```bash
cd lib
mkdir build && cd build
cmake ..
cmake --build .
```

Any changes to libmylib.so are automatically picked up each iteration
```bash
cd app
mkdir build && cd build
cmake ..
cmake --build .
./myapp
```