## install Conda
```
scp kyao@mlt-ace.sh.intel.com:/home2/kyao/Anaconda3-2021.05-Linux-x86_64.sh /home2/kyao
```
or
```
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.05-Linux-x86_64.sh
```
and then
```
bash Anaconda3-2021.05-Linux-x86_64.sh
```
## create Conda env
```
conda create --name stockPT python=3.6.13 
python --version
```
```
conda install gcc_linux-64
conda install -c anaconda libgcc-ng
conda install -c conda-forge gcc
conda install -c conda-forge gcc_linux-64
conda install gxx_linux-64
```
Note: you must do ```conda install gxx_linux-64``` otherwise you cannot do ```python setup.py develop```
```
conda install cmake
pip install --upgrade cmake==3.20.5
```
## inspect for right gcc/cmake source and version
```
which gcc
which cmake
which c++
which cpp
```
```
gcc --version
cmake --version
```
Make sure they're from conda path, if not (if path shows ```usr/bin/gcc``` ), add soft link:
```
ln -s /home/user_name/anconda3/envs/mmdeteciton/libexec/gcc/x8664=condacos6-linux-gnu/7.3.0/gcc /home/user_name/anconda3/envs/mmdeteciton/bin/gcc
ln -s /usr/bin/cpp /home2/kyao/anaconda3_this/envs/cc_env3613/bin/cpp
```
Also set up env var:
```
export LD_LIBRARY_PATH=/path/to/cmake:/path/to/gcc/lib64:/path/to/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/path/to/cmake/bin:/path/to/gcc/bin:/path/to/cuda/bin:$PATH
```
## install stockPT
```python setup.py develop``` instead of ```python setup.py install```  
If you get error and retry, do ```rm -rf build``` before you run ```python setup.py develop``` again  
Test if successful by ```python```,```import torch```

## install mkl if you get oserror:intel_mkllib when "import torch"
```
conda install mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home2/kyao/anaconda3_this/envs/cc_env3613/lib/
```
note1: use conda install instead of pip install, otherwise will not work  
note2: this error is machine dependent and may or may not occur  
## install torchvision with --no-deps
```
pip install --no-deps torchvision
```
Inspect torch and torchvision versions:
```
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
```