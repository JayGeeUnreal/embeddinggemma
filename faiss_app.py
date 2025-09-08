import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize
import torch

# --------------- Config ---------------
# Ensure MODEL_NAME is correct and the model is accessible (locally or on Hugging Face)
# If you're using a local model, replace this with the path to your model directory.
# Example: MODEL_NAME = "/path/to/your/local/gemma/model"
MODEL_NAME = "google/embeddinggemma-300m" # This is the Hugging Face model ID

# Corrected device detection and assignment
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
# --------------------------------------

queries = {
    # Top ~25 languages asking the same question
    "English": "Which planet is known as the Red Planet?",
    "Mandarin Chinese": "哪颗行星被称为红色星球?",
    "Hindi": "कौन सा ग्रह लाल ग्रह के रूप में जाना जाता है?",
    "Spanish": "¿Qué planeta es conocido como el Planeta Rojo?",
    "French": "Quelle planète est connue comme la planète rouge ?",
    "Arabic": "أي كوكب يعرف بالكوكب الأحمر؟",
    "Bengali": "কোন গ্রহটিকে লাল গ্রহ বলা হয়?",
    "Portuguese": "Qual planeta é conhecido como o Planeta Vermelho?",
    "Russian": "Какая планета известна как Красная планета?",
    "Urdu": "کون سا سیارہ سرخ سیار کے نام سے جانا جاتا ہے؟",
    "Indonesian": "Planet mana yang dikenal sebagai Planet Merah?",
    "German": "Welcher Planet ist als der Rote Planet bekannt?",
    "Japanese": "赤い惑星として知られているのはどの惑星ですか？",
    "Swahili": "Ni sayari ipi inajulikana kama Sayari Nyekundu?",
    "Marathi": "कोणता ग्रह लाल ग्रह म्हणून ओळखला जातो?",
    "Telugu": "ఏ గ్రహాన్ని అరుణ గ్రహం అంటారు?",
    "Turkish": "Hangi gezegen Kızıl Gezegen olarak bilinir?",
    "Tamil": "எந்த கிரகம் சிவப்புக்கோள் என அழைக்கப்படுகிறது?", # Note: This Tamil entry seems incomplete. You might want to fix it.
    "Italian": "Quale pianeta è conosciuto como il Pianeta Rosso?",
    "Korean": "어느 행성이 붉은 행성으로 알려져 있습니까?",
    "Vietnamese": "Hành tinh nào được gọi là Hành tinh Đỏ?",
    "Polish": "Która planeta jest znana jako Czerwona Planeta?",
    "Ukrainian": "Яка планета відома як Червона планета?",
    "Persian": " کدام سیاره به عنوان سیاره سرخ شناخته میشود؟",
    "Malay": "Planet mana yang dikenali sebagai Planet Merah?"
}

# Documents in English (but could also be multilingual)
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]

def main():
    # Load model
    print(f"Loading model {MODEL_NAME} on {DEVICE} ...")
    try:
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        torch.set_default_dtype(torch.float32) # ensure float32 activations
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        print("Please ensure you have accepted the Gemma license on Hugging Face and are logged in if required.")
        return

    # Encode documents into embeddings
    print("Encoding documents...")
    try:
        doc_embs = model.encode(documents, convert_to_numpy=True, device=DEVICE, show_progress_bar=False)
        # Normalize embeddings for cosine similarity
        doc_embs = normalize(doc_embs, axis=1).astype(np.float32)
    except Exception as e:
        print(f"Error encoding documents: {e}")
        return

    # Build FAISS GPU index (cosine similarity via inner product on normalized vectors)
    print("Building FAISS index...")
    try:
        d = doc_embs.shape[1] # Dimension of the embeddings
        # Initialize FAISS GPU resources
        res = faiss.StandardGpuResources()
        # Create an IndexFlatIP (Inner Product) index.
        # For normalized vectors, Inner Product is equivalent to Cosine Similarity.
        index = faiss.IndexFlatIP(d)
        # Move the index to the GPU
        index = faiss.index_cpu_to_gpu(res, 0, index)
        # Add the document embeddings to the index
        index.add(doc_embs)
        print(f"FAISS index built with {index.ntotal} vectors.")

        # --- Example: Search for similarity ---
        # You would typically encode a query and then search the index.
        # For example, let's search for documents similar to "Mars":
        query_text = "What is the Red Planet?"
        print(f"\nEncoding query: '{query_text}'")
        query_emb = model.encode([query_text], convert_to_numpy=True, device=DEVICE)
        query_emb = normalize(query_emb, axis=1).astype(np.float32)

        # Search the index
        k = 2 # Number of nearest neighbors to retrieve
        print(f"Searching for top {k} similar documents...")
        distances, indices = index.search(query_emb, k)

        print("\nSearch Results:")
        for i in range(k):
            print(f"  Rank {i+1}:")
            print(f"    Distance (Cosine Similarity): {distances[0][i]:.4f}")
            print(f"    Document: \"{documents[indices[0][i]]}\"")

    except Exception as e:
        print(f"Error building or using FAISS index: {e}")
        print("Ensure you have FAISS installed with GPU support (faiss-gpu) if you intend to use GPU acceleration.")

if __name__ == "__main__":
    main()