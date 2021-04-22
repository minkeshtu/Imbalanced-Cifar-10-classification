# Imbalanced-Cifar-10-classification
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
            (MobilenetV2 based: end to end training, Unet based: end to end training) 

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
