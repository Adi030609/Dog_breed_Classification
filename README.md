## Dog Classification with EfficientNetB0

This is a well-structured Python script for classifying dog breeds using EfficientNetB0 and Keras. This README file provides an overview of the code and its functionalities.

**1. Imports:**

- The code imports necessary libraries for image processing, deep learning, model building, evaluation, and visualization.

**2. Constants:**

- `IMG_SIZE`: Defines the image size for pre-processing (224x224 pixels).
- `NUM_CLASSES`: Specifies the number of dog breeds you aim to classify (120 in this case).
- `BATCH_SIZE`: Defines the batch size for training (32 images per batch).
- `EPOCHS`: Sets the number of training epochs (iterations) (30, can be increased for better accuracy).

**3. Paths:**

- `TRAIN_DIR`: Path to the directory containing your training data (dog breed images).
- `TEST_DIR`: Path to the directory containing your test data (dog breed images for evaluation).
- `WEIGHTS_PATH`: Path to the pre-trained EfficientNetB0 weights file (without the top classification layer).

**4. Data Augmentation and Normalization:**

- `train_datagen`: Defines an ImageDataGenerator for training data. This performs data augmentation techniques like rotation, shifting, zooming, flipping, etc., to increase the size and diversity of training data and prevent overfitting.
- `test_datagen`: Defines an ImageDataGenerator for test data. This only performs rescaling (normalization) to a range of 0-1 for consistency with the model's input.

**5. Loading and Augmenting Data:**

- `train_generator`: Uses `train_datagen` to load, augment, and prepare training data in batches for model training.
- `test_generator`: Uses `test_datagen` to load, rescale, and prepare test data in batches for model evaluation.

**6. Load Pre-trained Model:**

- `base_model`: Loads the pre-trained EfficientNetB0 model without the final classification layer (weights loaded from `WEIGHTS_PATH`). EfficientNetB0 is a powerful convolutional neural network (CNN) pre-trained on a large image dataset.

**7. Build the Model:**

- The model takes the output of the pre-trained EfficientNetB0 as input.
- A GlobalAveragePooling2D layer reduces the spatial dimensions of the features.
- A Dense layer with 256 units and ReLU activation adds non-linearity.
- BatchNormalization helps improve training stability.
- Dropout (0.3) helps prevent overfitting.
- A final Dense layer with `NUM_CLASSES` units and softmax activation provides the probabilities for each dog breed class.

**8. Compile the Model:**

- Adam optimizer with a learning rate of 0.0001 is used.
- Categorical cross-entropy loss is used for multi-class classification.
- Accuracy is used as the evaluation metric.

**9. Train the Model:**

- `history` stores the training and validation loss during each epoch.

**10. Plot Training and Validation Loss:**

- Visualizes the training and validation loss curves to monitor the learning process.

**11. Evaluate the Model:**

- Evaluates the model on the test data and prints the test loss and accuracy.

**12. Predict Classes:**

- Predicts the class probabilities for each image in the test data.
- Chooses the class with the highest probability as the predicted class.

**13. Generate Confusion Matrix:**

- Creates a confusion matrix to visualize the performance of the model on each dog breed class.

**14. Generate Classification Report:**

- Prints a classification report summarizing the model's performance, including precision, recall, F1-score, and support for each class.
