from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read text data
with open("data.txt", "r", encoding="utf-8") as file:
    documents = file.readlines()

# Create embeddings
embeddings = model.encode(documents)

# Convert to numpy array
embeddings = np.array(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings
index.add(embeddings)

print("Vector database created successfully!")