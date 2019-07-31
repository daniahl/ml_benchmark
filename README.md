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

## Sample Output
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