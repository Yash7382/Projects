import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from gensim.models import KeyedVectors

# Step 1: Load Word Embeddings (GloVe or Word2Vec)
# You can download pre-trained Word2Vec or GloVe embeddings.
# Here we use Gensim to load Word2Vec embeddings.
# Download Word2Vec embeddings from https://raw.githubusercontent.com/3Top/word2vec-api/master/data/GoogleNews-vectors-negative300.bin

model = KeyedVectors.load_word2vec_format('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz', binary=True)

# Step 2: Define the Fuzzy Logic Function
def fuzzy_similarity(word1, word2):
    """Calculates the fuzzy similarity between two words."""
    return fuzz.ratio(word1.lower(), word2.lower()) / 100.0

# Step 3: Prepare Data for Training
def get_synonyms(word):
    """Get a list of synonyms (for simplicity, use some predefined synonyms here)."""
    # You can use APIs or datasets like WordNet or pre-built lists for real-world applications.
    # For this example, using some predefined synonyms for illustration.
    predefined_synonyms = {
        "happy": ["joyful", "content", "cheerful", "pleased", "glad"],
        "sad": ["unhappy", "downcast", "mournful", "sorrowful", "depressed"],
        "fast": ["quick", "rapid", "speedy", "swift", "hasty"]
    }
    return predefined_synonyms.get(word, [])

# Step 4: Feature Extraction (Word Embedding + Fuzzy Logic)
def extract_features(word, synonym_list):
    word_vector = model[word]
    features = []
    for synonym in synonym_list:
        synonym_vector = model[synonym]
        cosine_sim = cosine_similarity([word_vector], [synonym_vector])[0][0]
        fuzzy_sim = fuzzy_similarity(word, synonym)
        features.append(np.array([cosine_sim, fuzzy_sim]))
    return np.array(features)

# Step 5: Neural Network Model
class SynonymNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SynonymNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Step 6: Training the Model
def train_model(model, criterion, optimizer, features, labels, epochs=100):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 7: Test the Model
def test_model(model, word, synonyms):
    model.eval()
    features = extract_features(word, synonyms)
    features_tensor = torch.tensor(features, dtype=torch.float32)
    output = model(features_tensor)
    predicted_class = torch.argmax(output).item()
    return synonyms[predicted_class]

# Step 8: Putting it All Together
if __name__ == "__main__":
    word = "happy"  # Input word to find synonyms for
    synonyms = get_synonyms(word)

    # Prepare data for training
    features = []
    labels = []
    synonym_list = get_synonyms(word)
    for i, synonym in enumerate(synonym_list):
        synonym_features = extract_features(word, [synonym])
        features.append(synonym_features)
        labels.append(i)  # Each synonym gets its own label

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)

    # Convert to tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Define the model, criterion, and optimizer
    input_size = 2  # Cosine similarity + fuzzy similarity
    hidden_size = 16
    output_size = len(synonym_list)

    model = SynonymNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, criterion, optimizer, features_tensor, labels_tensor)

    # Test the model
    predicted_synonym = test_model(model, word, synonym_list)
    print(f"The predicted synonym for '{word}' is: {predicted_synonym}")
