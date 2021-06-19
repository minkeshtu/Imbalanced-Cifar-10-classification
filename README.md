# Imbalanced-Cifar-10-classification

Implemented Popular techniques to tackle Imbalanced dataset. 

#### Methods Implemented to tackle imbalanced dataset
1. Data Over sampling:
    - In data over sampling we duplicate the samples from minority classes. 
    I used a weighted random sampling in which a batch is sampled according to the class weightage and class weights are calculated using inverse of class sample frequency of the dataset.
      ```
      Wn,c = 1/(no of samples in class c) 	where Wn,c is weight for class c
      ```
2. Class balanced loss:
    - It's a method (described in the paper: Class Balanced loss) to give class weightage in classification loss. This technique overcomes both problems of data oversampling and weighted loss using inverse of class sample frequency. Oversampling is prone to overfit whereas weighted loss does not take hard samples into account. In that paper, they designed a re-weighting scheme that uses the effective number of samples for each class to re-balance the loss, thereby yielding a class-balanced loss . The effective number of samples is defined as the volume of samples and can be calculated by a simple formula.
      ```
      Effective number of samples = a (1−β^n)/(1−β), 
      where n is number of samples, β ∈ [0, 1) is a hyperparameter

      Class balanced loss = 1/effective number of samples *loss
      where loss ∈  [Softmax, Sigmoid, Focal]

      ```

#### Network Architecture
<img src="/results_images/architecture.png" />

Auto encoder based classification models on Cifar-10 imbalanced dataset.

### Docker build and run

```
docker build -t <image name> .

docker run --runtime=nvidia -it --name <container_name> <image_name>
```

### Experiment results visualization

- Results of each experiment will be saved in 2 different directories.
  - experiment_reports directory:

    - A detailed experiment report (docx file) will be generated and a deployable model (encoder+classifier, after removing the decoder part) will be saved that is produced by that experiment.

      > I also included the experiment reports and deployable models of best performing scenarios
      > (MobilenetV2 based: end to end training, Unet based: end to end training)
      >
  - results directory: Here all the results will be saved regarding that experiment.

    - Best models of that experiment and checkpoints on last epoch/iteration
    - Tensorboard file (tf-event file) for visualizing training and validation over iterations
    - 2 txt files. evaluation results on test dataset, and hyperparameters of that experiment

### Examples

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
python3 train.py --is_classificaiton_training --arch unet_based --num_epochs 50
```
