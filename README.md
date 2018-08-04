# caffe2_cv
CV(CNN, GAN) based on caffe2, thanks to FengChao's https://zhuanlan.zhihu.com/hsmyy

## 1. Prepare pytorch/caffe2 on mac
- Source code preparation
```bash
git clone  --recursive https://github.com/pytorch/
git submodule update --init --recursive
git submodule update —remote
```
caffe2 is in the subfolder pytorch/caffe2

- Compilation preparation
1. AVX native support for speed-up  
Reference: https://github.com/caffe2/caffe2/issues/1789    
Open CMakeLists.txt, for line 113 change as follow
```bash
option(USE_NATIVE_ARCH "Use -march=native" ON)
```
2. CUDA libary linkage  
Reference: https://discuss.pytorch.org/u/llv22/activity    
Open setup.py, adding as follow
```bash
 main_libraries += ['cudart', 'nvToolsExt', 'nvrtc', 'cuda']
```

- Compilation
Reference:https://github.com/pytorch/pytorch#from-source    
```bash
export CMAKE_PREFIX_PATH=/Users/llv23/miniconda3
MACOSX_DEPLOYMENT_TARGET=10.13 CC=clang CXX=clang++ python setup.py install
```

- Linkage refinement (If caffe2_pybind11_state_gpu.cpython-36m-darwin.so not linked correctly) 
Reference:https://stackoverflow.com/questions/33991581/install-name-tool-to-update-a-executable-to-search-for-dylib-in-mac-os-x    

1. otool link check
```bash
otool -L <path to caffe2_pybind11_state_gpu.cpython-36m-darwin.so>
```
2. otool link refinement (For my case, 9.1->9.2)
```bash
install_name_tool -change @rpath/libcufft.9.1.dylib @rpath/libcufft.9.2.dylib <path to caffe2_pybind11_state_gpu.cpython-36m-darwin.so>
install_name_tool -change  @rpath/libcurand.9.1.dylib  @rpath/libcurand.9.2.dylib <path to caffe2_pybind11_state_gpu.cpython-36m-darwin.so>
install_name_tool -change @rpath/libnvrtc.9.1.dylib  @rpath/libnvrtc.9.2.dylib <path to caffe2_pybind11_state_gpu.cpython-36m-darwin.so>
install_name_tool -change @rpath/libcublas.9.1.dylib  @rpath/libcublas.9.2.dylib <path to caffe2_pybind11_state_gpu.cpython-36m-darwin.so>
```

- Test CPU/GPU for caffe2
Reference: https://caffe2.ai/docs/RNNs-and-LSTM-networks.html  
```bash
cd caffe2/python/examples
```
Download https://caffe2.ai/static/datasets/shakespeare.txt to that folder
  * CPU testing
  ```bash
  python char_rnn.py --train_data shakespeare.txt
  ```
  * GPU testing
  ```bash
  python char_rnn.py --train_data shakespeare.txt --gpu
  ```

