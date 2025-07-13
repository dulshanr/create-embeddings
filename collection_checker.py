from src.milvus_client import MilvusClient

def main():
    client = MilvusClient()
    if not client.connect():
        print("Failed to connect to Milvus.")
        return

    # This will load or create the collection named "pdf_documents" as per config.py
    if not client.create_collection():
        print("Failed to create or load collection.")
        return

    stats = client.get_collection_stats()
    print("Milvus Collection Stats for 'pdf_documents':")
    print(stats)

    client.close()

if __name__ == "__main__":
    main()