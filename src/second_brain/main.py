from second_brain.agents.ingestor import ingest_folder, query_notes, reset_collection
from second_brain.agents.thought_agent import run_thought_agent


def main():
    print("\n🧠 Welcome to your *Second Brain Assistant*!")
    print("============================================")

    while True:
        print("\n✨ Choose an option below:")
        print("1️⃣  Ingest all data into memory")
        print("2️⃣  Ask your Second Brain a question")
        print("3️⃣  Reset (delete) all stored data 🗑️")
        print("4️⃣  Test Thought Agent 💭")
        print("5️⃣  Exit ❌")

        choice = input("\n👉 Enter your choice (1-5): ").strip()

        if choice == "1":
            print("\n📂 Starting data ingestion...")
            ingest_folder()
            print("✅ All data has been successfully ingested!")
        
        elif choice == "2":
            query = input("\n🔍 What would you like to know? ")
            print("\n🧩 Searching through your knowledge base...\n")
            query_notes(query)
        
        elif choice == "3":
            confirm = input("⚠️  This will permanently delete all stored data. Type 'yes' to confirm: ").strip().lower()
            if confirm == "yes":
                reset_collection()
                print("🧹 ChromaDB collection has been reset successfully.")
            else:
                print("❌ Reset cancelled. Your data is safe!")
        
        elif choice == "4":
            user_prompt = input("\n💬 Enter your thought prompt: ")
            print("\n🤔 Thinking...")
            run_thought_agent(user_prompt)
        
        elif choice == "5":
            print("\n👋 Exiting Second Brain. See you next time!")
            break
        
        else:
            print("\n🚫 Invalid choice. Please enter a number between 1 and 5.")


if __name__ == "__main__":
    main()
