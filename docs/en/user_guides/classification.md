# Classification

In MMSelfSup, we provide many benchmarks, thus the models can be evaluated on different downstream tasks. Here are comprehensive tutorials and examples to explain how to run all benchmarks with MMSelfSup.

- [Classification](#classification)
  - [VOC SVM / Low-shot SVM](#voc-svm--low-shot-svm)
  - [Linear Evaluation](#linear-evaluation)
  - [ImageNet Semi-Supervised Classification](#imagenet-semi-supervised-classification)
  - [ImageNet Nearest-Neighbor Classification](#imagenet-nearest-neighbor-classification)

As for classification, we provide scripts in folder `tools/benchmarks/classification/`, which has 4 `.sh` files, 1 folder for VOC SVM related classification task and 1 folder for ImageNet nearest-neighbor classification task.

## VOC SVM / Low-shot SVM

To run these benchmarks, you should first prepare your VOC datasets. Please refer to [prepare_data.md](https://github.com/open-mmlab/mmselfsup/blob/master/docs/en/prepare_data.md) for the details of data preparation.

To evaluate the pre-trained models, you can run command below.

```shell
# distributed version
bash tools/benchmarks/classification/svm_voc07/dist_test_svm_pretrain.sh ${SELFSUP_CONFIG} ${GPUS} ${PRETRAIN} ${FEATURE_LIST}

# slurm version
bash tools/benchmarks/classification/svm_voc07/slurm_test_svm_pretrain.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${PRETRAIN} ${FEATURE_LIST}
```

Besides, if you want to evaluate the ckpt files saved by runner, you can run command below.

```shell
# distributed version
bash tools/benchmarks/classification/svm_voc07/dist_test_svm_epoch.sh ${SELFSUP_CONFIG} ${EPOCH} ${FEATURE_LIST}

# slurm version
bash tools/benchmarks/classification/svm_voc07/slurm_test_svm_epoch.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${EPOCH} ${FEATURE_LIST}
```

**To test with ckpt, the code uses the epoch\_\*.pth file, there is no need to extract weights.**

Remarks:

- `${SELFSUP_CONFIG}` is the config file of the self-supervised experiment.
- `${FEATURE_LIST}` is a string to specify features from layer1 to layer5 to evaluate; e.g., if you want to evaluate layer5 only, then `FEATURE_LIST` is "feat5", if you want to evaluate all features, then `FEATURE_LIST` is "feat1 feat2 feat3 feat4 feat5" (separated by space). If left empty, the default `FEATURE_LIST` is "feat5".
- `PRETRAIN`: the pre-trained model file.
- if you want to change GPU numbers, you could add `GPUS_PER_NODE=4 GPUS=4` at the beginning of the command.
- `EPOCH` is the epoch number of the ckpt that you want to test

## Linear Evaluation

The linear evaluation is one of the most general benchmarks, we integrate several papers' config settings, also including multi-head linear evaluation. We write classification model in our own codebase for the multi-head function, thus, to run linear evaluation, we still use `.sh` script to launch training. The supported datasets are **ImageNet**, **Places205** and **iNaturalist18**.

```shell
# distributed version
bash tools/benchmarks/classification/dist_train_linear.sh ${CONFIG} ${PRETRAIN}

# slurm version
bash tools/benchmarks/classification/slurm_train_linear.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${PRETRAIN}
```

Remarks:

- The default GPU number is 8. When changing GPUS, please also change `samples_per_gpu` in the config file accordingly to ensure the total batch size is 256.
- `CONFIG`: Use config files under `configs/benchmarks/classification/`. Specifically, `imagenet` (excluding `imagenet_*percent` folders), `places205` and `inaturalist2018`.
- `PRETRAIN`: the pre-trained model file.

## ImageNet Semi-Supervised Classification

To run ImageNet semi-supervised classification, we still use `.sh` script to launch training.

```shell
# distributed version
bash tools/benchmarks/classification/dist_train_semi.sh ${CONFIG} ${PRETRAIN}

# slurm version
bash tools/benchmarks/classification/slurm_train_semi.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${PRETRAIN}
```

Remarks:

- The default GPU number is 4.
- `CONFIG`: Use config files under `configs/benchmarks/classification/imagenet/`, named `imagenet_*percent` folders.
- `PRETRAIN`: the pre-trained model file.

## ImageNet Nearest-Neighbor Classification

To evaluate the pre-trained models using the nearest-neighbor benchmark, you can run command below.

```shell
# distributed version
bash tools/benchmarks/classification/knn_imagenet/dist_test_knn_pretrain.sh ${SELFSUP_CONFIG} ${PRETRAIN}

# slurm version
bash tools/benchmarks/classification/knn_imagenet/slurm_test_knn_pretrain.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${PRETRAIN}
```

Besides, if you want to evaluate the ckpt files saved by runner, you can run command below.

```shell
# distributed version
bash tools/benchmarks/classification/knn_imagenet/dist_test_knn_epoch.sh ${SELFSUP_CONFIG} ${EPOCH}

# slurm version
bash tools/benchmarks/classification/knn_imagenet/slurm_test_knn_epoch.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${EPOCH}
```

**To test with ckpt, the code uses the epoch\_\*.pth file, there is no need to extract weights.**

Remarks:

- `${SELFSUP_CONFIG}` is the config file of the self-supervised experiment.
- `PRETRAIN`: the pre-trained model file.
- if you want to change GPU numbers, you could add `GPUS_PER_NODE=4 GPUS=4` at the beginning of the command.
- `EPOCH` is the epoch number of the ckpt that you want to test