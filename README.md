# Imbalanced-Cifar-10-classification

Implemented Popular techniques to tackle Imbalanced dataset.

| Labels | No Class Balancing | Data Oversampling | Class Balanced loss - Focal | Class Balanced loss - Softmax | Clas Balanced loss - Sigmoid |
| :--: | :--: | :--: | :--: | :--: | :--: |
| <img src="/results_images/class_labels.png" /> **F-1**| <img src="/results_images/feature_clusters__no-class-balancing.png" /> **86.53 %** | <img src="/results_images/feature_clusters__data-oversampling.png" /> **83.92 %** | <img src="/results_images/feature_clusters__focal-class-balanced-loss.png" /> **84.19 %** | <img src="/results_images/feature_clusters__softmax-class-balancing-loss.png" /> **86.71 %** | <img src="/results_images/feature_clusters__sigmoid-class-balancing-loss.png" /> **87.48%** |

<p align="center">
<i>
Cluster visualization for Feature maps of 1000 images (from test data), and Macro F-1 of Classification Model on 10,000 test images. t-sne is used to reduce the dimentionality of feature vectors (made from feature maps)
</i>
</p>
 

## Methods Implemented to tackle imbalanced dataset
1. Data Over sampling:
    - In data over sampling we duplicate the samples from minority classes. 
    I used a weighted random sampling in which a batch is sampled according to the class weightage and class weights are calculated using inverse of class sample frequency of the dataset.
      ```
      Wn,c = 1/(no of samples in class c) 	where Wn,c is weight for class c
      ```
2. Class balanced loss:
    - It's a method (described in the paper: [Class Balanced loss](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)) to give class weightage in classification loss. This technique overcomes both problems of data oversampling and weighted loss using inverse of class sample frequency. Oversampling is prone to overfit whereas weighted loss does not take hard samples into account. In that paper, they designed a re-weighting scheme that uses the effective number of samples for each class to re-balance the loss, thereby yielding a class-balanced loss . The effective number of samples is defined as the volume of samples and can be calculated by a simple formula.
      ```
      Effective number of samples = a (1−β^n)/(1−β), 
      where n is number of samples, β ∈ [0, 1) is a hyperparameter

      Class balanced loss = 1/effective number of samples * loss
      where loss ∈  [Softmax, Sigmoid, Focal]

      ```

## Network Architecture
<img src="/results_images/architecture.png" />

Implemented Feature extractor as an encoder to extract features more precisely. See [this section](#effect-of-using-decoder-into-the-network) for comaprition of these networks with simple classficaiton models of the same network.

| # | Encoder | Decoder | Classifer
| - | ------- | ------- | ---------
| 1 | [Unet based](architectures/Unet_based.py#L34) | [Unet based](architectures/Unet_based.py#L58) | [Dense layer based](architectures/Unet_based.py#L94)
| 2 | [MobilenetV2 based](architectures/Mobilenetv2_based.py#L25) | [TransposeConv based](architectures/Mobilenetv2_based.py#L35) | [1x1 Conv based](architectures/Mobilenetv2_based.py#L59)

## Dataset
[Cifar-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- Training:	41650 -> 2450 samples  for ('truck', 'bird', 'deer') classes and 4900 samples for other classes
    - Although you can experiment with different imbalanced combination by just using the command line parameters
- Validation:	1,000 -> 100 samples in each class
- Test:		10,000 -> 1,000 samples in each class



## Results
[Unet based model](architectures/Unet_based.py) is used to produce all the below results. Encoder, Decorer, Classifier, all were trained together in a shared loss setting
#### F-1

| # | class Balancing Technique | Classification Loss | Macro F-1 (%)|
| - | -- | -- | -- |
| 1 | - | Categorical Cross Entropy | 86.53 % |
| 2 | Data Oversampling | Categorical Cross Entropy | 83.92 % |
| 3 | Class balancing | Focal Loss | 84.19 % |
| 4 | Class balancing | Softmax Loss | 86.71 % |
| 5 | Class balancing | Sigmoid Loss | 87.48 % |


#### Clusters visualization of feature maps (Encoder output)

| Labels | No Class Balancing (1) | Data Oversampling (2) | Class Balanced loss - Focal (3) | Class Balanced loss - Softmax (4) | Clas Balanced loss - Sigmoid (5) |
| :--: | :--: | :--: | :--: | :--: | :--: |
| <img src="/results_images/class_labels.png" /> | <img src="/results_images/feature_clusters__no-class-balancing.png" /> | <img src="/results_images/feature_clusters__data-oversampling.png" /> | <img src="/results_images/feature_clusters__focal-class-balanced-loss.png" /> | <img src="/results_images/feature_clusters__softmax-class-balancing-loss.png" /> | <img src="/results_images/feature_clusters__sigmoid-class-balancing-loss.png" /> |

<p align="center">
<i>
Cluster visualization for Feature maps of 1000 images (from test data). t-sne is used to reduce the dimentionality of feature vectors (made from feature maps)
</i>
</p>

## Effect of using Decoder into the Network

All the results are produced using the `Class balanced loss with Sigmoid` as a classification loss

| Architecture | Training type | Macro F-1 |
| :--: | :--: | :--: |
| Unet Based| End-to-end (Encoder+Decoder+Classifier) | 87.48% |
| Unet Based| Classification model only (Encoder+Classifier) | 87.03% |
| MobilenetV2 Based| End-to-end (Encoder+Decoder+Classifier) | 87.43% |
| MobilenetV2 Based| Classification model only (Encoder+Classifier) | 86.93% |

## Run with Docker

```
docker build -t <image name> .

docker run --gpus all -it -v <parent dir>/Imbalanced-Cifar-10-classfication:<docker dir to mount> --name <container_name> <image_name>
```

### Examples to run

End to end training of unet based model in a shared loss (autoender + classifier)

```
python3 train.py --is_end_to_end_training --arch unet_based --num_epochs 50
```

Separate training of Mobilenetv2 based model. First Auto encoder will be trained then classifer model will be trained after freezing the encoder model

```
python3 train.py --is_separate_training --arch mobilenetv2_based --num_epochs 50
```

Classification model training of Unet based architecture. Simple classification model training (Encoder + classifer, no decoder)

```
python3 train.py --is_classification_training --arch unet_based --num_epochs 50
```

### Experiment results visualization

- Results of each experiment will be saved in 2 different directories.
  - `experiment_reports` directory:

    - A detailed experiment report (docx file) will be generated and a deployable model (encoder+classifier, after removing the decoder part) will be saved that is produced by that experiment.

      > I also included the experiment reports and deployable models of best performing scenarios
      > (MobilenetV2 based: end to end training, Unet based: end to end training)
      >
  - `results` directory: Here all the results will be saved respective to the experiment.

    - Best models of the experiment and checkpoints on last epoch/iteration
    - Tensorboard file (tf-event file) for visualizing training and validation over iterations
    - 2 txt files. 1. evaluation matrices on test dataset, 2. hyperparameters of the experiment




