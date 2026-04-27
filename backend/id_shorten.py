import chromadb
import uuid

# 1. Connect to your local DB
client = chromadb.PersistentClient(path="D:/Abhin s/UG4/Sem 2/Applied Gen AI/Final_Project/backend/chroma_db")

# 2. Get the old collection
old_coll = client.get_collection(name="case_studies")
data = old_coll.get(include=['embeddings', 'documents', 'metadatas'])
total_items = len(data['ids'])
print(f"Total items found: {total_items}")

# 3. Create a NEW temporary collection
# If you already ran the script and it partially failed, you might need to delete it first:
try:
    client.delete_collection(name="case_studies_cleaned")
except:
    pass

new_coll = client.create_collection(name="case_studies_cleaned")

# 4. Add data in batches of 5000
batch_size = 5000
for i in range(0, total_items, batch_size):
    end_idx = min(i + batch_size, total_items)
    
    new_coll.add(
        ids=[str(uuid.uuid4()) for _ in range(i, end_idx)],
        embeddings=data['embeddings'][i:end_idx],
        documents=data['documents'][i:end_idx],
        metadatas=data['metadatas'][i:end_idx]
    )
    print(f"Successfully processed items {i} to {end_idx}...")

print("Finished! New collection 'case_studies_cleaned' is ready.")