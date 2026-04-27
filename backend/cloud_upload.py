import chromadb

# --- 1. Get your local data ---
print("Reading local database...")
local_client = chromadb.PersistentClient(path="D:/Abhin s/UG4/Sem 2/Applied Gen AI/Final_Project/backend/chroma_db")
local_coll = local_client.get_collection("case_studies_cleaned")
data = local_coll.get(include=['embeddings', 'documents', 'metadatas'])
total = len(data['ids'])

# --- 2. Connect to Cloud exactly as shown in your screenshot ---
print("Connecting to Chroma Cloud...")
cloud_client = chromadb.CloudClient(
    api_key='ck-CYRXghxNxzsib72mF3JjPwXJG1mLLHnnS4oohn71c9LW',
    tenant='0ce0bd1c-6617-44ba-a01c-734cf27d25b1',
    database='brand_decision_db'
)

# --- 3. Create the collection ---
cloud_coll = cloud_client.get_or_create_collection(name="case_studies")

# --- 4. Upload the data ---
print(f"Migrating {total} records. Watch your Chroma Cloud dashboard...")
batch_size = 100

for i in range(0, total, batch_size):
    end_idx = min(i + batch_size, total)
    cloud_coll.add(
        ids=data['ids'][i:end_idx],
        embeddings=data['embeddings'][i:end_idx],
        documents=data['documents'][i:end_idx],
        metadatas=data['metadatas'][i:end_idx]
    )
    if (i // batch_size) % 10 == 0:
        print(f"Uploaded {end_idx} / {total}...")

print("\n✨ Done! Check your browser.")