import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from gensim.models import KeyedVectors

def load_dataset(path):
    """
    Lee un archivo CSV y divide los datos en entrenamiento y prueba.
    """
    df = pd.read_csv(path)
    df = df[['review_en', 'sentiment']]
    df.columns = ['review_en', 'sentiment']
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return train_test_split(df['review'], df['label'], test_size=0.2, random_state=42)

def load_embeddings(path):
    """
    Carga un modelo Word2Vec preentrenado en formato binario.
    """
    return KeyedVectors.load_word2vec_format(path, binary=True)

def vectorize_reviews(reviews, w2v):
    """
    Convierte cada review a un vector promedio de sus palabras.
    """
    vectors = []
    for review in reviews:
        words = review.lower().split()
        word_vecs = [w2v[word] for word in words if word in w2v]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(w2v.vector_size))
    return np.array(vectors)