from chatbot_function import DataProcessor, Chatbot

def chat():
    data_processor = DataProcessor()
    chatbot = Chatbot(data_processor)
    print("Medical Chatbot is ready! Type 'exit' to stop.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! Take care!")
            break
        best_question, response, source = chatbot.get_response(user_input)
        if best_question == "UNKNOWN":
            print(f"\nChatbot: {response}")
        else:
            print(f"\nChatbot (matched question): {best_question}")
            print(f"Chatbot (answer): {response}")
            print(f"Source: {source}")

if __name__ == "__main__":
    chat()
