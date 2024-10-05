# Bias Detection NLP
Text Bias Detection and Debiasing
This project implements a system for detecting and mitigating bias in text using various machine learning models. The pipeline includes a TF-IDF + Logistic Regression classifier for bias detection and a T5 model for text debiasing. Additionally, BERT is used for masked language modeling and generating alternative phrasings.
Requirements
	•	torch
	•	transformers
	•	nltk
	•	pandas
	•	scikit-learn
	•	time
You can install the required packages using pip:
bash
pip install torch transformers nltk pandas scikit-learn
Code Overview
	1	Data Loading and Preprocessing
	◦	Load and preprocess data from train.csv.
	◦	Split data into training and testing sets.
	2	Bias Detection
	◦	A TF-IDF vectorizer combined with Logistic Regression is used to classify text into biased or non-biased categories.
	3	Bias Prediction
	◦	The predict_bias function predicts the level of bias in a given text.
	4	Text Debiasing
	◦	The T5 model generates debiased versions of the input text.
	◦	The debias_text function uses the T5 model to rephrase biased text.
	5	Alternative Phrasing Generation
	◦	BERT's masked language model is used to generate alternative phrasings for biased text.
	◦	The generate_alternatives function provides multiple alternative phrasings.
	6	Model Inference
	◦	A fine-tuned BERT model is used for inference, providing class predictions and probabilities.
Usage
	1	Load and Train Model
	◦	Ensure train.csv is available in the /content/ directory.
	◦	The model is trained and evaluated on this data.
	2	Test Predictions and Debiasing
	◦	Use the predict_bias and debias_text functions to test bias detection and debiasing on sample texts.
	3	Generate Alternatives
	◦	Use the generate_alternatives function to get alternative phrasings for biased texts.
	4	Model Inference
	◦	Use bert_inference to get predictions from a fine-tuned BERT model.

Notes
	•	Ensure NLTK data is downloaded (punkt and averaged_perceptron_tagger).
	•	Modify file paths and model names as needed.
