# Audio_Classification_MFCC

Audio Classification using Mel Frequency Cepstral Coefficients (MFCCs) and Random Forest Classifier

This code is an implementation of a Random Forest Classifier for audio classification using MFCCs (Mel Frequency Cepstral Coefficients) as features.

1. Import the necessary libraries: The code imports various libraries, including librosa for audio feature extraction, numpy for numerical operations, pandas for handling data in tabular format, matplotlib for visualization, and scikit-learn for machine learning.

2. Define the function to extract MFCCs: The `extract_mfccs` function takes an audio file path as input, loads the audio waveform using librosa, and then calculates the MFCCs with 13 coefficients for each audio file. The function returns the mean of the MFCC coefficients along the columns, effectively reducing the dimensionality to a 1D array.

3. Load and preprocess audio data: The code loads the audio files of dolphin and non-dolphin sounds and extracts the MFCCs for each. The extracted MFCCs are stored in a DataFrame, along with their corresponding labels ('dolphin' or 'non-dolphin'). The DataFrame is then saved to a CSV file named 'Baph_MFCCs.csv'.

4. Visualize the waveform and MFCCs: The code plots the waveform and MFCCs of the dolphin and non-dolphin audio files using matplotlib. This provides a visual representation of the audio features.

5. Load the dataset and split into training and testing sets: The code loads the previously saved CSV file containing the MFCC features and labels. The features (MFCCs) are stored in the variable `X`, and the labels are stored in the variable `y`. The labels are converted into integers using the `LabelEncoder`. Then, the dataset is split into training and testing sets with a 70-30 split.

6. Train a Random Forest Classifier: The code initializes a Random Forest Classifier and trains it using the training data (`X_train` and `y_train`).

7. Make predictions and evaluate the model: The trained Random Forest Classifier is used to make predictions on the test data (`X_test`). The accuracy of the model is calculated using the `accuracy_score` function from scikit-learn. Additionally, the confusion matrix is calculated using the `confusion_matrix` function to evaluate the performance of the model.

Note: The accuracy and performance of the model depend on the size and quality of the dataset and the audio features used for classification. In this code, with only one instance per class, the accuracy may not be meaningful, and the model may not generalize well to new, unseen data. To build a more robust model, you need a larger and more balanced dataset with sufficient instances for each class. Additionally, feature engineering and tuning of the classifier's hyperparameters can further improve the model's performance.
