# ai bad bags model

this project is a convolutional neural network (cnn) based classifier designed to detect good and bad bread bags from images. it includes full data preprocessing, model training, evaluation, and visualization steps.

## project structure

```
AI_model_bread_images/
│── dataset/                # original dataset of bread bag images
│── dataset_split/          # auto-generated stratified split (train/val/test)
│── models/                 # saved trained models
│── plots/                  # training history and metrics plots
│── raw_video/              # source code directory
│   ├── train_classifier.py # main training script
│   ├── augment_bad_bags.py # script to augment bad bag images
│── README.md               # project documentation
│── requirements.txt        # dependencies
```

## dataset

- total images: ~450
- classes: 
  - **good_bags**: properly sealed bread bags
  - **bad_bags**: defective or unsealed bread bags
- data split:
  - 80% training
  - 10% validation
  - 10% test
- preprocessing:
  - images resized to 128x128
  - normalized pixel values (0–1)

## models

this project tests and compares multiple cnn architectures:

1. **baseline model** – simple cnn with 2–3 convolutional layers
2. **deeper model** – at least 4 convolutional layers, batch normalization, dropout
3. **alt pooling model** – alternative pooling strategies
4. **transfer learning model** – pretrained network fine-tuned for classification
5. **balanced model** – uses focal loss and oversampling for class imbalance

### key features:
- focal loss for handling imbalance
- data augmentation (rotation, flipping, zooming, shifting)
- class weights for emphasizing minority class

## results

- best threshold for balanced model: **0.65**
- final balanced model:
  - precision: 0.93
  - recall: 1.00
  - f1-score: 0.97
- overall accuracy: 96–98% depending on model
- transfer learning model performed best with ~98% accuracy

confusion matrix and misclassified examples are available in the `plots/` folder.

## how to run

1. clone the repo:
   ```bash
   git clone https://github.com/davidjbergman/AI_BAD_BAGS_MODEL.git
   ```
2. create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. train the model:
   ```bash
   python3 raw_video/train_classifier.py
   ```
5. saved models will be available in the `models/` folder.

## visualization

- training and validation accuracy/loss plots
- class distribution and data augmentation previews
- precision, recall, f1-score charts
- confusion matrix with error analysis

## future improvements

- collect more bad bag images for improved recall
- try other advanced architectures (e.g., efficientnet)
- deploy as a real-time inference api or edge device solution
- add automated mlflow experiment tracking

---

created by david bergman
