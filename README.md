Text Classification & Feature Analysis with Naive Bayes

This project is a comprehensive data science study that explores the internal dynamics of the Naive Bayes algorithm, the impact of feature selection, and performance differences between n-gram models in text classification tasks. Rather than focusing solely on prediction, it seeks mathematical answers to the question: “Which words indicate a class, and why?”

Project Features

Beyond a standard fit-predict workflow, the project includes the following analyses:
	•	Distinctiveness Scoring:
A custom metric is developed to measure how specific a word is to a particular class. This enables the extraction of signature words for each category (e.g., Sports, Politics).
	•	N-Gram Analysis:
Unigram (1-gram) and Bigram (2-gram) models are built to evaluate how contextual information affects classification performance.
	•	Absence Effect:
The effect of a word not appearing in a document on its probability of belonging to a certain class is analyzed
(e.g., the absence of the word “dollar” may reduce the likelihood that the text belongs to the Economy category).
	•	Stop-Word Impact:
The behavioral differences of the model when using and not using the English Stop Words list are examined.

File Structure
	•	main.py: Main script containing data preprocessing, model training, and analysis functions.
	•	English Dataset.csv: Categorized text dataset used for training and testing.

Methodology
	1.	Data Preprocessing:
Texts are converted to lowercase (lower()) and punctuation is removed.
	2.	Vectorization:
Texts are transformed into numerical matrices (Bag of Words) using CountVectorizer.
	3.	Statistical Analysis:
	•	Word frequency distributions across classes are computed.
	•	Word importance scores are derived from log-probability differences.
