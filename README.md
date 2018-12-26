
# Tensorflow examples for C/C++
This project includes source code for my blog:

1. [Tensorflow C: from training to serving](http://www.liuxiao.org/2018/12/tensorflow-c-api-%E4%BB%8E%E8%AE%AD%E7%BB%83%E5%88%B0%E9%83%A8%E7%BD%B2%EF%BC%9A%E4%BD%BF%E7%94%A8-c-api-%E8%BF%9B%E8%A1%8C%E9%A2%84%E6%B5%8B%E5%92%8C%E9%83%A8%E7%BD%B2/) (In Chinese)

2. [Tensorflow C++: from training to serving](http://www.liuxiao.org/2018/08/ubuntu-tensorflow-c-%e4%bb%8e%e8%ae%ad%e7%bb%83%e5%88%b0%e9%a2%84%e6%b5%8b1%ef%bc%9a%e7%8e%af%e5%a2%83%e6%90%ad%e5%bb%ba/) (In Chinese)

# 1. Dependancies
1) For C API, download sdk from [https://www.tensorflow.org/install/lang_c](https://www.tensorflow.org/install/lang_c) and put it into third_party/tensorflow_cpu (default) or third_party/tensorflow_gpu folder.
 
2) For C++ API, follow the steps in [Tensorflow C++: from training to serving](http://www.liuxiao.org/2018/08/ubuntu-tensorflow-c-%e4%bb%8e%e8%ae%ad%e7%bb%83%e5%88%b0%e9%a2%84%e6%b5%8b1%ef%bc%9a%e7%8e%af%e5%a2%83%e6%90%ad%e5%bb%ba/) (In Chinese) or [Tensorflow C++ API](https://www.tensorflow.org/guide/extend/cc) to build tensorflow on your platform.

The C++ API is only designed to work with TensorFlow bazel build, which means you have to build tensorflow on every devices. If you need a stand-alone option, we suggest you use the C API. 

# 2. Build
## Prepare

```shell
mkdir build
cd build
```

## CMake

For C API, run:
 ```shell
cmake .. -DUSE_TENSORFLOW_C=ON
```

For C++ API, run:
 ```shell
cmake .. -DUSE_TENSORFLOW_CPP=ON
```

## Build

 ```shell
make
```

# 3. Run
## For C++ Example (Simple Net)
```shell
./load_simple_net ../simple/model/simple.pb
```
## For C++ Example (CNN Net)
```shell
./load_predict_cnn ../fashion_mnist/models/fashion_mnist.h5.pb ../fashion_mnist/fashion_0.png
```
## For C Example (Simple Net)
```shell
./load_simple_net_c_api ../simple/model/simple.pb
```
## For C Example (CNN Net)
```shell
./load_predict_cnn_c_api ../fashion_mnist/models/fashion_mnist.h5.pb ../fashion_mnist/fashion_0.png
```
# Contact
My Blog: [www.liuxiao.org](http://www.liuxiao.org)

If you have any question, please contact liuxiao at foxmail.com
