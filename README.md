
# Tensorflow examples for C/C++
This project includes source code for my blog:

1. [Tensorflow C++: from training to serving](http://www.liuxiao.org/2018/08/ubuntu-tensorflow-c-%e4%bb%8e%e8%ae%ad%e7%bb%83%e5%88%b0%e9%a2%84%e6%b5%8b1%ef%bc%9a%e7%8e%af%e5%a2%83%e6%90%ad%e5%bb%ba/) (In Chinese)

# Dependancies
1. For C++ API

2. For C API, download sdk from [https://www.tensorflow.org/install/lang_c](https://www.tensorflow.org/install/lang_c) and put it into third_party/tensorflow_cpu (default) or third_party/tensorflow_gpu folder.
 
# Build
## Prepare

```shell
mkdir build
cd build
```

## CMake

For C++ API, run :
 ```shell
cmake .. -DUSE_TENSORFLOW_C=ON
```

For C++ API, run t:
 ```shell
cmake .. -DUSE_TENSORFLOW_CPP=ON
```

## Build

 ```shell
make
```

# Run
## For C++ Example (Simple Net)
```shell

```
## For C++ Example (CNN Net)
```shell

```
## For C Example (Simple Net)
```shell
./load_simple_net_c_api ../simple/model/simple.pb
```
## For C Example (CNN Net)
```shell
./load_predict_cnn_c_api ../fashion_mnist/models/fashion_mnist.h5.pb ../fashion_mnist/fashion_0.png
```

My Blog: [www.liuxiao.org](http://www.liuxiao.org)
