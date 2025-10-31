from second_brain.utils import setup_otel

# Initialize OpenTelemetry and logfire instrumentation (must be before other imports)
setup_otel()

# Import after OTEL/logfire setup to ensure instrumentation works
from second_brain.agents.ingestor import RAGManager
from second_brain.agents.thought_agent import ThoughtAgent

def main():
    print("\n🧠 Welcome to your *Second Brain Assistant*!")
    print("============================================")

    agent = ThoughtAgent()
    rag_manager = RAGManager()

    while True:
        print("\n✨ Choose an option below:")
        print("1️⃣  Ingest all data into memory")
        print("2️⃣  Ask your Second Brain a question")
        print("3️⃣  Reset (delete) all stored data 🗑️")
        print("4️⃣  Test Thought Agent 💭")
        print("5️⃣  Clear Memory 🧠")
        print("6️⃣  Exit ❌")

        choice = input("\n👉 Enter your choice (1-6): ").strip()

        if choice == "1":
            print("\n📂 Starting data ingestion...")
            rag_manager.ingest_folder()
            print("✅ All data has been successfully ingested!")

        elif choice == "2":
            query = input("\n🔍 What would you like to know? ")
            rag_manager.query_notes(query)

        elif choice == "3":
            confirm = input("⚠️ This will permanently delete all stored data. Type 'yes' to confirm: ").strip().lower()
            if confirm == "yes":
                rag_manager.reset_collection()
                print("🧹 ChromaDB collection has been reset successfully.")
            else:
                print("❌ Reset cancelled. Your data is safe!")

        elif choice == "4":
            user_prompt = input("\n💬 Enter your thought prompt: ")
            response = agent.run(user_prompt)
            print("\n🧠 Thought Agent Response:\n")
            print(response)
            print("\n" + "=" * 60 + "\n")

        elif choice == "5":
            confirm = input("⚠️ This will delete all memory. Type 'yes' to confirm: ").strip().lower()
            if confirm == "yes":
                agent.clear_memory()
            else:
                print("❌ Memory not cleared.")

        elif choice == "6":
            print("\n👋 Exiting Second Brain. See you next time!")
            break

        else:
            print("\n🚫 Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main()
