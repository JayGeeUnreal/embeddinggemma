import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import torch

# --------------- Config ---------------
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

    print("\n" + "="*30)
    print("  Multilingual Query Results")
    print("="*30)

    # --- Iterate through queries and perform search ---
    for lang, query_text in queries.items():
        print(f"\n[{lang}] {query_text}")
        try:
            # Encode the query
            query_emb = model.encode([query_text], convert_to_numpy=True, device=DEVICE)
            query_emb = normalize(query_emb, axis=1).astype(np.float32)

            # Calculate cosine similarity between the query embedding and all document embeddings
            similarities = cosine_similarity(query_emb, doc_embs)[0]

            # Get the indices of the top k most similar documents
            k = 2 # Number of nearest neighbors to retrieve

            # Ensure k is not greater than the number of documents
            k = min(k, len(documents))

            top_k_indices = np.argsort(similarities)[-k:][::-1]

            # Print top results for this query
            if len(top_k_indices) > 0:
                for i in range(k):
                    idx = top_k_indices[i]
                    print(f"  -> Top result {i+1}: '{documents[idx]}' (score={similarities[idx]:.3f})")
            else:
                print("  No similar documents found.")

        except Exception as e:
            print(f"  Error processing query for {lang}: {e}")

if __name__ == "__main__":
    main()
