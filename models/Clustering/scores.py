import tokenizer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def get_vector(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()

def get_score(label):
    if label == 0:
        return 1
    elif label == 2:
        return -1
    return 0
def get_clusters(terms, labels):
    categories = ["service", "food", "price", "ambience"]
    categories_scores = [0, 0, 0, 0]
    category_vectors = [get_vector(cat) for cat in categories]
    word_vectors = [get_vector(word) for word in terms]

    similarity_threshold = 0.7
    category_assignments = []

    for idx, word_vec in enumerate(word_vectors):
        similarities = [cosine_similarity(word_vec, cat_vec) for cat_vec in category_vectors]
        max_similarity = max(similarities)
        if max_similarity >= similarity_threshold:
            category_index = similarities.index(max_similarity)
            category_assignments.append(categories[category_index])
            print(labels[idx])
            categories_scores[category_index] += get_score(labels[idx])
        else:
            category_assignments.append(None)
    return categories_scores