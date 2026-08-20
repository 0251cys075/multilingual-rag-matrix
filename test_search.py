from app.retriever import search
import pprint

print("Loading ultra-fast search engine...")
_, _, _ = search("warmup", top_k=1)
print("Ready!\n")

while True:
    query = input("Enter a search query (or type 'quit' to exit): ")
    if query.lower() == 'quit':
        break
    
    results, embed_ms, search_ms = search(query, top_k=3)
    
    total_time = embed_ms + search_ms
    print(f"\n⏱️  Total Time: {total_time:.2f} ms")
    print("=" * 60)
    
    for i, res in enumerate(results):
        score = res.pop("score", 0.0)
        print(f"Result {i+1} | Distance Score: {score:.4f}")
        
        # Pretty print all fields (sentences, translations, language, etc.)
        for key, val in res.items():
            print(f"  [{key}]: {val}")
        print("-" * 60)
    
    print("\n")