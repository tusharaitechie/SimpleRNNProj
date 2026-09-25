<div align="center">

# 🎬 SimpleRNN Movie Review Sentiment Analysis

### End-to-End NLP + Deep Learning + Streamlit Deployment

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NLP](https://img.shields.io/badge/Domain-NLP-6A5ACD)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Deep Learning](https://img.shields.io/badge/Model-SimpleRNN-0B7285)](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN)

**Predict whether a movie review is Positive or Negative using a trained SimpleRNN model.**

<br>

### 🚀 [Try the Live Application](https://simplernnproj-ktfwqugcduyazamkz2zpom.streamlit.app/) &nbsp; | &nbsp; 💻 [View Source Code](https://github.com/tusharaitechie/SimpleRNNProj)

</div>

---

## 📌 Project Snapshot

**SimpleRNNProj** is an end-to-end **Natural Language Processing (NLP)** project for **binary sentiment classification** of movie reviews.

The project takes a raw text review, converts it into a numerical sequence using the **IMDB vocabulary**, pads the sequence to a fixed length, sends it through a trained **SimpleRNN** model, and presents the prediction through an interactive **Streamlit** application.

This repository demonstrates the complete path from **deep-learning experimentation to a deployable inference application**.

| Area | Implementation |
|---|---|
| Problem | Movie Review Sentiment Classification |
| Task | Binary Classification |
| Domain | Natural Language Processing |
| Model | Simple Recurrent Neural Network (SimpleRNN) |
| Framework | TensorFlow / Keras |
| Text Representation | IMDB Word Index |
| Sequence Length | 500 tokens |
| Interface | Streamlit |
| Deployment | Streamlit App |
| Model Artifact | `simple_rnn_imdb.h5` |

---

## 🎯 Problem Statement

Movie reviews contain useful information about a user's opinion, but raw text cannot be directly consumed by a neural network.

The goal of this project is to build an inference pipeline that transforms a review such as:

> **"The story was engaging and the performances were excellent."**

into a machine-readable sequence and predicts its sentiment as:

**Positive** or **Negative**.

---

## 🚀 Live Demo

### Try the application

👉 **[Open the Live Streamlit App](https://simplernnproj-ktfwqugcduyazamkz2zpom.streamlit.app/)**

The application provides a simple interface where a user can:

1. Enter a movie review.
2. Click **Classify**.
3. Receive the predicted sentiment.
4. View the model's prediction score.

---

## 🧠 Solution Architecture

```text
                    ┌──────────────────────┐
                    │     User Review       │
                    │   Raw Text Input      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Text Processing    │
                    │ lowercase + split    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   IMDB Word Index    │
                    │ Words → Integer IDs  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Sequence Padding     │
                    │     maxlen = 500     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Trained SimpleRNN  │
                    │   Keras Model (.h5)  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Prediction Score     │
                    │      0 → 1           │
                    └──────────┬───────────┘
                               │
                       ┌───────┴────────┐
                       ▼                ▼
                 🟢 POSITIVE       🔴 NEGATIVE
```

---

## 🔄 End-to-End Prediction Pipeline

The deployed application follows a compact and reproducible inference workflow:

### 1. Load the IMDB vocabulary

The application obtains the IMDB dataset word index and creates a reverse mapping for decoding.

### 2. Preprocess the review

The user's input is converted to lowercase and split into words.

### 3. Encode words

Each word is mapped to its corresponding IMDB integer ID. Unknown words are assigned an unknown-token representation.

### 4. Pad the sequence

The encoded review is padded to a fixed sequence length of **500**.

### 5. Load the trained model

The saved Keras model:

```text
simple_rnn_imdb.h5
```

is loaded for inference.

### 6. Generate prediction

The model produces a numerical prediction score.

### 7. Convert score to sentiment

The application uses a threshold of **0.5**:

```text
prediction > 0.5  → Positive
prediction ≤ 0.5  → Negative
```

### 8. Display the result

Streamlit displays both the sentiment and prediction score.

---

## 🧩 Core Implementation

The deployed inference logic is intentionally simple and easy to follow.

### Load the IMDB vocabulary and model

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
model = load_model("simple_rnn_imdb.h5")
```

### Convert text into a padded sequence

```python
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )
    return padded_review
```

### Generate the prediction

```python
prediction = model.predict(preprocessed_input)

sentiment = (
    "Positive"
    if prediction[0][0] > 0.5
    else "Negative"
)
```

The actual deployed implementation follows this same inference flow. citeturn1view0

---

## 🏗️ Model & NLP Concepts

This project provides hands-on exposure to several important Deep Learning and NLP concepts.

### Natural Language Processing

- Text preprocessing
- Vocabulary mapping
- Token-to-integer representation
- Sequence encoding
- Sequence padding
- Text classification

### Deep Learning

- Neural network based text classification
- Recurrent Neural Networks
- SimpleRNN
- Model inference
- Saved Keras model loading

### Deployment

- Streamlit application development
- Model serving
- Interactive prediction
- Cloud deployment

---

## 🔍 Why SimpleRNN?

Movie reviews are sequential text, where the order of words can influence meaning.

For example:

```text
"The movie was good."
```

and

```text
"The movie was not good."
```

contain many of the same words but communicate different sentiment.

A recurrent architecture is useful to study this type of sequential language information because the network processes the input as a sequence rather than treating the entire review as an unordered collection of words.

---

## 📂 Repository Structure

```text
SimpleRNNProj/
│
├── 📓 simplernn.ipynb
│   └── SimpleRNN model development / experimentation
│
├── 📓 embedding.ipynb
│   └── Embedding-related experimentation
│
├── 📓 prediction.ipynb
│   └── Prediction / inference experimentation
│
├── 🐍 main.py
│   └── Streamlit application + inference pipeline
│
├── 🤖 simple_rnn_imdb.h5
│   └── Trained Keras SimpleRNN model
│
├── 📦 requirements.txt
│   └── Python dependencies
│
└── 📄 README.md
    └── Project documentation
```

The current GitHub repository contains the three notebooks, `main.py`, the trained `.h5` model, and `requirements.txt`. citeturn0view0

---

## 🛠️ Technology Stack

| Technology | Role |
|---|---|
| **Python** | Application and ML programming |
| **TensorFlow / Keras** | Deep Learning framework |
| **SimpleRNN** | Recurrent neural network architecture |
| **NumPy** | Numerical operations |
| **Pandas** | Data manipulation / analysis |
| **Scikit-learn** | Machine Learning utilities |
| **TensorBoard** | Training / experiment monitoring |
| **Matplotlib** | Visualization |
| **SciKeras** | Keras + scikit-learn integration |
| **Streamlit** | Interactive ML web application |
| **Jupyter Notebook** | Model development and experimentation |

These dependencies are reflected in the repository's current `requirements.txt`. citeturn1view1

---

## 💻 Run the Project Locally

### Prerequisites

- Python 3.x
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/tusharaitechie/SimpleRNNProj.git
cd SimpleRNNProj
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run main.py
```

### 5. Open the local application

Streamlit will provide a local URL in the terminal. Open it in your browser and enter a movie review.

---

## 🧪 Example Inputs

### Positive

```text
The movie was fantastic, entertaining and beautifully acted.
```

Expected sentiment:

```text
Positive
```

### Negative

```text
The movie was boring and poorly written.
```

Expected sentiment:

```text
Negative
```

> The output is generated by the trained model and can vary depending on the wording and context of the review.

---

## 📈 Project Workflow

```text
                ┌─────────────────┐
                │   IMDB Dataset  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Text Processing │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Representation  │
                │ / Embedding     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    SimpleRNN    │
                │ Model Training  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Saved .h5 Model │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Streamlit App   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Live Prediction │
                └─────────────────┘
```

---

## 💼 Portfolio / Recruiter Highlights

### End-to-End Ownership

The project goes beyond notebook experimentation and includes a working application layer for model inference.

### Practical NLP Use Case

Sentiment classification is a widely applicable NLP problem with use cases such as:

- Customer feedback analysis
- Product review classification
- Brand sentiment monitoring
- Opinion mining
- Social media text analysis

### Model-to-Application Integration

The trained neural-network model is loaded directly by the Streamlit application, demonstrating the transition from a trained ML artifact to an interactive user-facing application.

### Reproducible Project Structure

The repository includes notebooks for experimentation, the trained model artifact, application code, and dependency specification.

---

## 🌐 Project Resources

| Resource | Link |
|---|---|
| 🚀 **Live Demo** | [Launch Streamlit Application](https://simplernnproj-ktfwqugcduyazamkz2zpom.streamlit.app/) |
| 💻 **GitHub Repository** | [View Source Code](https://github.com/tusharaitechie/SimpleRNNProj) |

---

## 🔮 Possible Future Enhancements

The current project can be extended with:

- Bidirectional RNN / LSTM / GRU comparison
- Pre-trained word embeddings
- Transformer-based sentiment classification
- Better text normalization and tokenization
- Model evaluation metrics dashboard
- Confidence visualization
- Batch review prediction
- Model performance comparison
- Docker-based deployment
- Automated CI/CD deployment

---

## 👨‍💻 Author

### Tushar

**Aspiring Data Scientist | Machine Learning Engineer | AI Engineer**

Areas of interest:

`Machine Learning` • `Deep Learning` • `NLP` • `Generative AI` • `AI Engineering`

---

<div align="center">

### ⭐ Explore the project

**[🚀 Live Demo](https://simplernnproj-ktfwqugcduyazamkz2zpom.streamlit.app/) · [💻 GitHub](https://github.com/tusharaitechie/SimpleRNNProj)**

</div>
