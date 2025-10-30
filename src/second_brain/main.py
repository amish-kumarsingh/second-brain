from second_brain.agents.ingestor import ingest_folder, query_notes, reset_collection

def main():
    print("🧠 Welcome to your Second Brain CLI")

    while True:
        print("\nChoose an option:")
        print("1️⃣  Ingest all data")
        print("2️⃣  Query notes")
        print("3️⃣  Reset ChromaDB collection")
        print("4️⃣  Exit")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            ingest_folder()
        elif choice == "2":
            query = input("🔍 Enter your query: ")
            query_notes(query)
        elif choice == "3":
            confirm = input("⚠️ This will delete all stored data. Type 'yes' to confirm: ").strip().lower()
            if confirm == "yes":
                reset_collection()
            else:
                print("❌ Reset cancelled.")
        elif choice == "4":
            print("👋 Exiting Second Brain. Goodbye!")
            break
        else:
            print("❌ Invalid choice. Try again.")

if __name__ == "__main__":
    main()
