# Sentiment Analysis with LSTM

This project implements a sentiment analysis model using LSTM (Long Short-Term Memory) networks built with TensorFlow and Keras. The model is trained on the IMDb movie reviews dataset to classify reviews as either positive or negative.

## Files

- **main.py**: The main script that trains and evaluates the sentiment analysis model.
- **test_model.py**: A helper script that allows users to input any text and test the model for sentiment prediction.

## Setup

### Prerequisites

Make sure you have Python 3.x and the following libraries installed:

- TensorFlow
- NumPy
- TensorFlow Datasets

You can install the required libraries by running:

```bash
pip install tensorflow numpy tensorflow-datasets
```

### Running the model

1. **Training the model**: 
   - Run the `main.py` script to load the IMDb dataset, preprocess the data, build and train the LSTM-based sentiment analysis model.
   
   ```bash
   python main.py
   ```

2. **Testing the model**:
   - After training, you can use the `test_model.py` script to test the model with your own sentences. This allows you to input any review and predict its sentiment.
   
   ```bash
   python test_model.py
   ```

### Model Details

The model consists of:
- An **Embedding layer** for word vectorization.
- Two **LSTM layers** for sequential data processing.
- A **Dense layer** with ReLU activation for non-linearity.
- A **final Dense layer** with a sigmoid activation function for binary classification (positive/negative).

### Preprocessing

- The IMDb dataset is loaded using TensorFlow Datasets.
- Texts are tokenized into word indices, and sequences are padded to ensure uniform input length.
- The vocabulary is created by counting word frequencies in the training data, and words not found in the vocabulary are assigned to an OOV (Out Of Vocabulary) token.

### Example Usage

After running `test_model.py`, you will be able to input any text and receive a sentiment prediction. Example:

```bash
python test_model.py
Enter text to predict sentiment: I really enjoyed the movie, it was amazing!
Prediction: Positive

Enter text to predict sentiment: This was the worst film I've ever seen.
Prediction: Negative
```

### Saving and Loading the Model

The trained model is saved in the file `sentiment_analysis_bot.keras`. You can reload the model using:

```python
from tensorflow.keras.models import load_model
model = load_model('sentiment_analysis_bot.keras')
```

## License

This project is open-source and available under the [MIT License](LICENSE).
