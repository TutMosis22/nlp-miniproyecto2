# Importamos funciones propias del proyecto
from src.preprocess import load_dataset, vectorize_reviews, load_embeddings
from src.model import FFNClassifier
from src.train import train_model

# Paso 1: Cargar el dataset
print("Cargando dataset...")
X_train, X_test, y_train, y_test = load_dataset("data/reviews.csv")

# Paso 2: Cargar embeddings Word2Vec preentrenados
print("Cargando embeddings Word2Vec...")
w2v = load_embeddings("glove-wiki-gigaword-100")

# Paso 3: Convertir cada review a un vector usando los embeddings
# Promediamos los vectores de cada palabra contenida en la review
print("Vectorizando reseñas...")
X_train_vec = vectorize_reviews(X_train, w2v)
X_test_vec = vectorize_reviews(X_test, w2v)

# Paso 4: Crear modelo de red neuronal feedforward
print("Inicializando modelo...")
model = FFNClassifier(input_dim=100)

# Paso 5: Entrenar modelo
print("Entrenando modelo...")
train_model(model, X_train_vec, y_train, X_test_vec, y_test)
