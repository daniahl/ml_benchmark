# ml_benchmark
Simple Python script to to test the performance of NumPy, scikit-learn, and tensorflow.

## Requirements
* Python >= 3.4
* NumPy
* tqdm

The above are minimum requirements, with which the script will run NumPy tests.

The other tests require:
* scikit-learn
* tensorflow (CPU or GPU)

## The Tests
**NumPy**: random matrix generation, matrix multiplication, singular value decomposition

**Scikit-learn**: classification dataset generation, SVM training, random forest training (both single and multithreaded)

**Tensorflow**: image dataset generation, ResNet50 training

## Anaconda Quick Start
1. `git clone https://github.com/daniahl/ml_benchmark.git`
2. `cd ml_benchmark`
3. `conda create -n ml_benchmark python=3.6`
4. `conda activate ml_benchmark`
5. `conda install scikit-learn tqdm tensorflow` (or `tensorflow-gpu`)
6. `python ml_benchmark`

To delete the environment:
1. `conda deactivate ml_benchmark`
2. `conda remove -n ml_benchmark --all`

## Sample Output
### MacOS
The output below was generated on a late 2015 iMac (Core i5-6600) running MacOS Mojave:

    mkl_info:
        libraries = ['mkl_rt', 'pthread']
        library_dirs = ['/Users/daniel/anaconda3/lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['/Users/daniel/anaconda3/include']
    blas_mkl_info:
        libraries = ['mkl_rt', 'pthread']
        library_dirs = ['/Users/daniel/anaconda3/lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['/Users/daniel/anaconda3/include']
    blas_opt_info:
        libraries = ['mkl_rt', 'pthread']
        library_dirs = ['/Users/daniel/anaconda3/lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['/Users/daniel/anaconda3/include']
    lapack_mkl_info:
        libraries = ['mkl_rt', 'pthread']
        library_dirs = ['/Users/daniel/anaconda3/lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['/Users/daniel/anaconda3/include']
    lapack_opt_info:
        libraries = ['mkl_rt', 'pthread']
        library_dirs = ['/Users/daniel/anaconda3/lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['/Users/daniel/anaconda3/include']

    Generating matrices...
    It took 0.37 s.

    Running matrix multiplication test...
    100%|█████████████████████████████████████████████| 5/5 [00:03<00:00,  1.36it/s]
    4096x4096 matrix multiplication in 0.74 s.

    Running SVD test...
    100%|█████████████████████████████████████████████| 5/5 [00:02<00:00,  2.37it/s]
    SVD of 2048x1024 matrix in 0.42 s.

    Scikit-learn version: 0.21.2

    Creating classification dataset (10 classes, 10000 samples, 50 features)...
    It took 0.02 s.

    Running SVM test...
    100%|█████████████████████████████████████████████| 5/5 [00:40<00:00,  8.22s/it]
    SVM trained in 8.17 s.
    Score on training set: 0.8351

    Running random forest test (1 thread)...
    100%|█████████████████████████████████████████████| 5/5 [00:26<00:00,  5.37s/it]
    Random forest trained in 5.36 s.
    Score on training set: 1.0

    Running random forest test (2 threads)...
    100%|█████████████████████████████████████████████| 5/5 [00:14<00:00,  2.96s/it]
    Random forest trained in 2.95 s.
    Score on training set: 1.0

    Running random forest test (max threads)...
    100%|█████████████████████████████████████████████| 5/5 [00:08<00:00,  1.70s/it]
    Random forest trained in 1.70 s.
    Score on training set: 1.0

    Tensorflow version: 1.13.1
    2019-07-31 09:41:01.487674: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
    No GPU found. Using CPU.

    Generating image data...
    It took 0.03 s.

    2019-07-31 09:41:04.773662: I tensorflow/core/common_runtime/process_util.cc:71] Creating new thread pool with default inter op setting: 4. Tune using inter_op_parallelism_threads for best performance.
    Training 5 epochs of 10-class ResNet50 on (20, 224, 224, 3)            ...
    Epoch 1/5
    20/20 [==============================] - 27s 1s/sample - loss: 2.8836 - acc: 0.0500
    Epoch 2/5
    20/20 [==============================] - 9s 473ms/sample -     loss: 2.3550 - acc: 0.1000
    Epoch 3/5
    20/20 [==============================] - 10s 495ms/sample - loss: 1.6474 - acc: 0.4000
    Epoch 4/5
    20/20 [==============================] - 11s 533ms/sample - loss: 1.5367 - acc: 0.4500
    Epoch 5/5
    20/20 [==============================] - 10s 484ms/sample - loss: 0.6262 - acc: 0.8000
    Training took 78.92 s.

### Windows
Specs: Core i7-9700k, 32GB RAM, NVIDIA Geforce GTX 1660 Ti, Windows 10 build 1903.

    NumPy version: 1.16.5

    mkl_info:
        libraries = ['mkl_rt']
        library_dirs = ['C:/Users/Daniel/Miniconda3/envs/ml_benchmark\\Library\\lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'C:/Users/Daniel/Miniconda3/envs/ml_benchmark\\Library\\include']
    blas_mkl_info:
        libraries = ['mkl_rt']
        library_dirs = ['C:/Users/Daniel/Miniconda3/envs/ml_benchmark\\Library\\lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'C:/Users/Daniel/Miniconda3/envs/ml_benchmark\\Library\\include']
    blas_opt_info:
        libraries = ['mkl_rt']
        library_dirs = ['C:/Users/Daniel/Miniconda3/envs/ml_benchmark\\Library\\lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'C:/Users/Daniel/Miniconda3/envs/ml_benchmark\\Library\\include']
    lapack_mkl_info:
        libraries = ['mkl_rt']
        library_dirs = ['C:/Users/Daniel/Miniconda3/envs/ml_benchmark\\Library\\lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'C:/Users/Daniel/Miniconda3/envs/ml_benchmark\\Library\\include']
    lapack_opt_info:
        libraries = ['mkl_rt']
        library_dirs = ['C:/Users/Daniel/Miniconda3/envs/ml_benchmark\\Library\\lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'C:/Users/Daniel/Miniconda3/envs/ml_benchmark\\Library\\include']

    Generating matrices...
    It took 0.24 s.

    Running matrix multiplication test...
    100%|████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.53it/s]
    4096x4096 matrix multiplication in 0.40 s.

    Running SVD test...
    100%|████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  4.49it/s]
    SVD of 2048x1024 matrix in 0.23 s.

    Scikit-learn version: 0.21.2

    Creating classification dataset (10 classes, 10000 samples, 50 features)...
    It took 0.02 s.

    Running SVM test...
    100%|████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:35<00:00,  7.09s/it]
    SVM trained in 7.10 s.
    Score on training set: 0.8351

    Running random forest test (1 thread)...
    100%|████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:22<00:00,  4.52s/it]
    Random forest trained in 4.52 s.
    Score on training set: 1.0

    Running random forest test (2 threads)...
    100%|████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:12<00:00,  2.40s/it]
    Random forest trained in 2.41 s.
    Score on training set: 1.0

    Running random forest test (max threads)...
    100%|████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:04<00:00,  1.24it/s]
    Random forest trained in 0.80 s.
    Score on training set: 1.0

    Tensorflow version: 1.14.0
    2019-09-15 09:20:42.754384: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
    2019-09-15 09:20:42.766034: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library nvcuda.dll
    2019-09-15 09:20:42.845834: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 0 with properties:
    name: GeForce GTX 1660 Ti major: 7 minor: 5 memoryClockRate(GHz): 1.815
    pciBusID: 0000:01:00.0
    2019-09-15 09:20:42.858303: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
    2019-09-15 09:20:42.869061: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1763] Adding visible gpu devices: 0
    2019-09-15 09:20:43.326715: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1181] Device interconnect StreamExecutor with strength 1 edge matrix:
    2019-09-15 09:20:43.335326: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1187]      0
    2019-09-15 09:20:43.340941: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 0:   N
    2019-09-15 09:20:43.347126: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/device:GPU:0 with 4637 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1660 Ti, pci bus id: 0000:01:00.0, compute capability: 7.5)
    WARNING: Logging before flag parsing goes to stderr.
    W0915 09:20:43.362314  1432 deprecation_wrapper.py:119] From ml_benchmark.py:109: The name tf.logging.set_verbosity is deprecated. Please use tf.compat.v1.logging.set_verbosity instead.

    W0915 09:20:43.363311  1432 deprecation_wrapper.py:119] From ml_benchmark.py:109: The name tf.logging.ERROR is deprecated. Please use tf.compat.v1.logging.ERROR instead.


    Generating image data...
    It took 0.02 s.

    2019-09-15 09:20:46.810171: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 0 with properties:
    name: GeForce GTX 1660 Ti major: 7 minor: 5 memoryClockRate(GHz): 1.815
    pciBusID: 0000:01:00.0
    2019-09-15 09:20:46.822910: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
    2019-09-15 09:20:46.832820: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1763] Adding visible gpu devices: 0
    2019-09-15 09:20:46.839365: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1181] Device interconnect StreamExecutor with strength 1 edge matrix:
    2019-09-15 09:20:46.848424: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1187]      0
    2019-09-15 09:20:46.853773: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 0:   N
    2019-09-15 09:20:46.859738: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4637 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1660 Ti, pci bus id: 0000:01:00.0, compute capability: 7.5)
    Training 5 epochs of 10-class ResNet50 on (20, 224, 224, 3)...
    Epoch 1/5
    20/20 [==============================] - 5s 273ms/sample - loss: 2.8183 - acc: 0.1000
    Epoch 2/5
    20/20 [==============================] - 1s 29ms/sample - loss: 2.4544 - acc: 0.2500
    Epoch 3/5
    20/20 [==============================] - 1s 29ms/sample - loss: 2.0335 - acc: 0.2000
    Epoch 4/5
    20/20 [==============================] - 1s 30ms/sample - loss: 1.7796 - acc: 0.4000
    Epoch 5/5
    20/20 [==============================] - 1s 29ms/sample - loss: 1.1573 - acc: 0.6000
    Training took 14.28 s.
