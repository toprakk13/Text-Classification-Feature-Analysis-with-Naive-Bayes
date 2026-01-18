import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy import sparse
df=pd.read_csv("English Dataset.csv")
df.duplicated().sum() 
df.isnull().sum()
df['Text'] = df['Text'].str.lower()
vectorizer = CountVectorizer(stop_words='english')  #remove stopwords
X = vectorizer.fit_transform(df['Text'])
words = vectorizer.get_feature_names_out()
categories = sorted(df['Category'].unique())
category_word_counts = {}  #array of word counts aligned with vocab

for i in categories:
    idx = df[df['Category'] == i].index
    counts = np.array(X[idx].sum(axis=0)).flatten()
    category_word_counts[i] = counts
word_scores = {}
for i, word in enumerate(words):
    freqs = np.array([category_word_counts[cat][i] for cat in categories])
    # the most frequent category
    max_idx = freqs.argmax()
    max_val = freqs[max_idx]
    #mean of other categories
    others_mean = (freqs.sum() - max_val) / max(1, (len(freqs)-1))
    score = max_val - others_mean
    word_scores[word] = (score, categories[max_idx], int(max_val))
top20 = sorted(word_scores.items(), key=lambda x: x[1][0], reverse=True)[:20]
keywords = [w for w,meta in top20[:3]]
print(keywords)
rows = []
for n in keywords:
    idx = vectorizer.vocabulary_.get(n, None)
    counts = {}
    for cat in categories:
        if idx is None:
            c = 0
        else:
            c = int(category_word_counts[cat][idx])
        counts[cat] = c
    rows.append({'keyword': n, **counts})

df_keywords = pd.DataFrame(rows)
df_keywords = df_keywords[['keyword'] + categories]  
print(df_keywords.to_string(index=False))




#naive bayes class
class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None    # log P(y)
        self.feature_log_prob_ = None   # log P(w|y)
        self.V = None

    def fit(self, X, y):
        # X: (n_samples, n_features) sparse or dense
        if sparse.issparse(X):
            X = X.tocsr()
        else:
            X = np.asarray(X)
        self.classes_, counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.V = n_features

        # log prior
        self.class_log_prior_ = np.log(counts) - np.log(y.shape[0])

        # feature count per class
        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)
        for idx, c in enumerate(self.classes_):
            mask = (y == c)
            if sparse.issparse(X):
                feature_count[idx, :] = X[mask].sum(axis=0)
            else:
                feature_count[idx, :] = X[mask].sum(axis=0)

        # total words per class
        class_total = feature_count.sum(axis=1)  # shape (n_classes,)

        # Laplace smoothing and log probabilities
        denom = (class_total + self.alpha * self.V).reshape(-1,1)  # (n_classes,1)
        self.feature_log_prob_ = np.log((feature_count + self.alpha) / denom)  # (n_classes, n_features)
        return self

    def predict_log_proba(self, X):
        if sparse.issparse(X):
            jll = X.dot(self.feature_log_prob_.T)  # (n_samples, n_classes)
        else:
            jll = np.dot(X, self.feature_log_prob_.T)
        jll = jll + self.class_log_prior_
        return jll

    def predict(self, X):
        logp = self.predict_log_proba(X)
        idx = np.argmax(logp, axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)


# 2) Pipeline function: vectorize (ngram), split, train, evaluate
def run_pipeline(df, ngram_range=(1,1), alpha=1.0, test_size=0.2, random_state=42, remove_stopwords=True):
    texts = df['Text'].fillna('').values
    labels = df['Category'].values

    # stratified split for the distribution of each category
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    stop_words = 'english' if remove_stopwords else None
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)

    # train model
    mnb = MultinomialNaiveBayes(alpha=alpha)
    mnb.fit(X_train, y_train)

    # metrics
    train_acc = mnb.score(X_train, y_train)
    test_acc = mnb.score(X_test, y_test)

    print(f"ngram_range={ngram_range}, alpha={alpha}, stopwords_removed={remove_stopwords}")
    print(f"Train accuracy: {train_acc*100:.2f}%")
    print(f"Test  accuracy: {test_acc*100:.2f}%")

    
    return {
        'vectorizer': vectorizer,
        'model': mnb,
        'X_train_texts': X_train_texts,
        'X_test_texts': X_test_texts,
        'y_train': y_train,
        'y_test': y_test,
        'X_train': X_train,
        'X_test': X_test
    }


# 3) Feature effect

def top_k_words_per_class(model, vectorizer, k=10):
    
    inv_vocab = np.array(vectorizer.get_feature_names_out())
    classes = model.classes_
    results = {}
    for i, c in enumerate(classes):
        # feature_log_prob_[i] : log P(w|c)
        top_idx = np.argsort(model.feature_log_prob_[i])[-k:][::-1]
        words = inv_vocab[top_idx]
        logps = model.feature_log_prob_[i][top_idx]
        results[c] = list(zip(words, logps))
    return results

def top_k_absence(model, vectorizer, k=10):
    
    inv_vocab = np.array(vectorizer.get_feature_names_out())
    classes = model.classes_
    n_classes = len(classes)
    logP = model.feature_log_prob_  # (n_classes, n_features)
    results = {}
    for i, c in enumerate(classes):
        # compute mean log P(w|c) for other classes
        others_idx = [j for j in range(n_classes) if j != i]
        others_mean = logP[others_idx].mean(axis=0)
        # relative difference of log probabilities
        diff = logP[i] - others_mean  # # positive -> strong presence; negative ->strong absence
        # select k most negative words (absence signal)
        top_idx = np.argsort(diff)[:k]  
        words = inv_vocab[top_idx]
        results[c] = list(zip(words, diff[top_idx]))
    return results


# most important words
results_unigram = run_pipeline(df, ngram_range=(1,1), alpha=1.0)
top_words = top_k_words_per_class(results_unigram['model'], results_unigram['vectorizer'], k=10)

for cat, words in top_words.items():
    print(f"\nCategory: {cat}")
    for w, logp in words:
        print(f"{w}: {logp:.3f}")

# absence effect
absence_effects = top_k_absence(results_unigram['model'], results_unigram['vectorizer'], k=10)

for cat, words in absence_effects.items():
    print(f"\nCategory (absence effect): {cat}")
    for w, diff in words:
        print(f"{w}: {diff:.3f}")
results_with_stopwords = run_pipeline(df, ngram_range=(1,1), remove_stopwords=False)
results_without_stopwords = run_pipeline(df, ngram_range=(1,1), remove_stopwords=True)



results_biagram = run_pipeline(df, ngram_range=(2,2), alpha=1.0)
top_biagrams = top_k_words_per_class(results_biagram['model'], results_biagram['vectorizer'], k=10)

for cat, words in top_biagrams.items():
    print(f"\nCategory: {cat}")
    for w, logp in words:
        print(f"{w}: {logp:.3f}")


absence_effects_biagram = top_k_absence(results_biagram['model'], results_biagram['vectorizer'], k=10)

for cat, words in absence_effects_biagram.items():
    print(f"\nCategory (absence effect): {cat}")
    for w, diff in words:
        print(f"{w}: {diff:.3f}")
#biagram

results_bigram_without_stopwords = run_pipeline(df, ngram_range=(2,2), alpha=1.0, remove_stopwords=True)
results_bigram_with_stopwords = run_pipeline(df, ngram_range=(2,2), alpha=1.0, remove_stopwords=False)
