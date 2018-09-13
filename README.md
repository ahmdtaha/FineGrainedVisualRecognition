# FGVR Tensorflow Baseline
Fine grained visual recognition (FGVR) tensorflow baseline.


| Dataset       | Num Classes | Avg samples Per Class | Train Size | Val Size | Test Size |
|---------------|-------------|-----------------------|------------|----------|-----------|
| Flowers-102   | 102         | 10                    | 1020       | 1020     | 6149      |
| CUB-200-2011  | 200         | 29.97                 | 5994       | N/A      | 5794      |
| Cars          | 196         | 41.55                 | 8144       | N/A      | 8041      |
| NABirds       | 550         | 43.5                  | 23929      | N/A      | 24633     |
| Aircrafts     | 100         | 100                   | 3334       | 3333     | 3333      |
| Stanford Dogs | 120         | 100                   | 12000      | N/A      | 8580      |

Requirements

* Python 3+ [Tested on 3.4.7]
* Tensorflow 1+ [Tested on 1.8]

Running

Make sure to edit the (1) dataset\_dir (2) pretrained\_weights\_dir in configuration.py.
Run `python fgvr_train.py `



Credits:

* [Accumulated Gradient in Tensorflow.](https://stackoverflow.com/questions/46772685/how-to-accumulate-gradients-in-tensorflow)
* [Training and Evaluating at both time](https://github.com/tensorflow/tensorflow/issues/5987).
* [Batch Level Augmentation](https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19).
* [Pretrained-DenseNet](https://github.com/pudae/tensorflow-densenet)

I hope I didnt forget other people


