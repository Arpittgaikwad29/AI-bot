import os
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = "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"

class TherapistAI:
    def __init__(self):
        self.setup_nlp_components()
        self.setup_voice_components()
        self.conversation = []
        self.is_listening = False
        self.listening_thread = None

    def setup_nlp_components(self):
        # Initialize embedding model
        print("🔄 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load the persisted database
        print("📁 Loading pre-calculated vector database...")
        persist_directory = "chroma_db"
        self.vectorstore = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever()

        # Initialize Groq LLM
        print("🤖 Initializing LLM...")
        groq_api_key = "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"
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

        # Define system prompt
        system_prompt = (
            "You are a compassionate and knowledgeable AI therapist. "
            "Your role is to provide emotional support, practical guidance, and mental health insights "
            "based on the retrieved context. Respond with empathy and validation while offering helpful solutions. "
            "Your responses should be concise (3-5 sentences) and action-oriented.\n\n"
            "Balance these approaches in your responses:\n"
            "1. Brief validation of feelings (1 sentence)\n"
            "2. One relevant insight or perspective shift\n"
            "3. One practical technique or exercise when appropriate (breathing exercises, grounding techniques, etc.)\n\n"
            "Use the following retrieved context to answer the question. "
            "If needed, ask one focused follow-up question, but prioritize providing immediate value. "
            "Your responses should be warm yet direct, creating a supportive space without unnecessary length.\n\n"
            "{context}"
        )

        # Set up prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            ("system", "Use the following context: {context}")
        ])

        # Initialize chat history
        self.chat_history = ChatMessageHistory()

        # Create LLM chain
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

        # Create StuffDocumentsChain for document processing
        self.stuff_chain = StuffDocumentsChain(
            llm_chain=self.llm_chain,
            document_variable_name="context",
            input_key="input_documents"
        )

    def setup_voice_components(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        
        # Initialize text-to-speech with explicit initialization
        try:
            print("🔊 Setting up text-to-speech engine...")
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
            
            # Get available voices
            voices = self.engine.getProperty('voices')
            print(f"Available voices: {len(voices)}")
            
            # Try to set a female voice if available
            female_voice_set = False
            for voice in voices:
                print(f"Found voice: {voice.name} ({voice.id})")
                if 'female' in voice.name.lower():
                    print(f"Setting female voice: {voice.name}")
                    self.engine.setProperty('voice', voice.id)
                    female_voice_set = True
                    break
                    
            if not female_voice_set and voices:
                print(f"Using default voice: {voices[0].name}")
                self.engine.setProperty('voice', voices[0].id)
                
            # Test speech engine
            print("Testing text-to-speech...")
            self.engine.say("Voice system initialized")
            self.engine.runAndWait()
            print("Text-to-speech test completed")
            
        except Exception as e:
            print(f"⚠️ Error initializing text-to-speech: {str(e)}")
            self.engine = None

    def chat_with_ai(self, user_input):
        """Process user input and generate AI response with RAG"""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.invoke(user_input)
            print(f"\n🔍 Retrieved {len(retrieved_docs)} documents")
            
            if retrieved_docs:
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            else:
                context = "No specific context available. Responding based on general knowledge."
            
            # Prepare input data with history and context
            input_data = {
                "input": user_input,
                "context": context,
                "history": self.chat_history.messages,
                "input_documents": retrieved_docs 
            }
            
            # Generate response
            response = self.stuff_chain.invoke(input_data)
            
            # Extract response text with proper error handling
            if isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            elif isinstance(response, dict) and "output_text" in response:
                response_text = response["output_text"]
            elif isinstance(response, str):
                response_text = response
            else:
                print("⚠️ Unexpected response format:", response)
                response_text = "I'm here to listen and support you. Could you tell me more about what's on your mind?"
            
            # Add the interaction to chat history
            self.chat_history.add_user_message(user_input)
            self.chat_history.add_ai_message(response_text)
            
            return response_text
        
        except Exception as e:
            print(f"⚠️ Error: {str(e)}")
            # Fallback response in case of errors
            fallback_response = "I'm here to support you. Can you share more about how you're feeling today?"
            self.chat_history.add_user_message(user_input)
            self.chat_history.add_ai_message(fallback_response)
            return fallback_response

    def listen_for_speech(self):
        """Listen for speech and convert to text"""
        with sr.Microphone() as source:
            self.gui.update_status("Listening... Speak now.")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                self.gui.update_status("Processing speech...")
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.WaitTimeoutError:
                self.gui.update_status("No speech detected. Try again.")
                return None
            except sr.UnknownValueError:
                self.gui.update_status("Sorry, I couldn't understand that.")
                return None
            except Exception as e:
                self.gui.update_status(f"Error: {str(e)}")
                return None

    def start_listening(self):
        """Start listening for voice input in a separate thread"""
        if not self.is_listening:
            self.is_listening = True
            self.listening_thread = threading.Thread(target=self._listen_thread)
            self.listening_thread.daemon = True
            self.listening_thread.start()

    def _listen_thread(self):
        """Thread function for voice input"""
        while self.is_listening:
            text = self.listen_for_speech()
            if text:
                self.gui.receive_voice_input(text)
                self.is_listening = False  # Stop listening after receiving input
                break
            else:
                self.is_listening = False

    def stop_listening(self):
        """Stop listening for voice input"""
        self.is_listening = False
        self.gui.update_status("Listening stopped.")

    def speak_text(self, text):
        """Convert text to speech with robust error handling"""
        if not text:
            print("Nothing to speak")
            return
            
        if self.engine is None:
            print("⚠️ Text-to-speech engine not available")
            return
            
        try:
            print(f"🔊 Speaking: {text[:50]}...")
            self.gui.update_status("Speaking...")
            
            # Create a new thread for speaking
            def speak_thread():
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                    self.gui.update_status("Ready")
                except Exception as e:
                    print(f"⚠️ Error in speak thread: {str(e)}")
                    self.gui.update_status("Speech error")
            
            # Run the speaking in a separate thread
            speech_thread = threading.Thread(target=speak_thread)
            speech_thread.daemon = True
            speech_thread.start()
            
        except Exception as e:
            print(f"⚠️ Error initiating speech: {str(e)}")
            self.gui.update_status("Speech error")

    def set_gui(self, gui):
        """Set the GUI reference"""
        self.gui = gui


class TherapistAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Therapist AI Assistant")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        self.therapist = TherapistAI()
        self.therapist.set_gui(self)
        
        self.setup_gui_components()
        
    def setup_gui_components(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text="Therapeutic AI Assistant", font=("Arial", 18, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Status indicator
        self.status_label = ttk.Label(title_frame, text="Ready", font=("Arial", 10), foreground="blue")
        self.status_label.pack(side=tk.RIGHT)
        
        # Conversation display
        conversation_frame = ttk.LabelFrame(main_frame, text="Conversation", padding=10)
        conversation_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.conversation_text = scrolledtext.ScrolledText(conversation_frame, wrap=tk.WORD, font=("Arial", 11))
        self.conversation_text.pack(fill=tk.BOTH, expand=True)
        self.conversation_text.config(state=tk.DISABLED)
        
        # User input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.input_text = ttk.Entry(input_frame, font=("Arial", 11))
        self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_text.bind("<Return>", lambda event: self.send_message())
        
        send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        send_button.pack(side=tk.RIGHT, padx=(0, 5))
        
        # Voice controls frame
        voice_frame = ttk.Frame(main_frame)
        voice_frame.pack(fill=tk.X)
        
        voice_input_button = ttk.Button(voice_frame, text="🎤 Voice Input", command=self.therapist.start_listening)
        voice_input_button.pack(side=tk.LEFT)
        
        self.voice_output_var = tk.BooleanVar(value=True)
        voice_output_check = ttk.Checkbutton(voice_frame, text="Voice Output", variable=self.voice_output_var)
        voice_output_check.pack(side=tk.LEFT, padx=10)
        
        # Test voice button
        test_voice_button = ttk.Button(voice_frame, text="Test Voice", command=self.test_voice)
        test_voice_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = ttk.Button(voice_frame, text="Clear Chat", command=self.clear_conversation)
        clear_button.pack(side=tk.RIGHT)
        
        # Add some initial welcome message
        self.add_message("assistant", "Hello! I'm your therapeutic AI assistant. How are you feeling today?")
        
    def test_voice(self):
        """Test the text-to-speech functionality"""
        self.update_status("Testing voice output...")
        threading.Thread(target=self.therapist.speak_text, 
                        args=("This is a test of the voice output system.",), 
                        daemon=True).start()
        
    def send_message(self):
        """Send the user's message to the therapist AI"""
        user_input = self.input_text.get().strip()
        if not user_input:
            return
            
        self.input_text.delete(0, tk.END)
        self.process_user_input(user_input)
        
    def receive_voice_input(self, text):
        """Process voice input received from the speech recognition"""
        self.update_status("Voice input received.")
        self.process_user_input(text)
        
    def process_user_input(self, user_input):
        """Process the user input and get AI response"""
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            self.add_message("user", user_input)
            self.add_message("assistant", "Goodbye! Take care. 💙")
            return
            
        # Display user message
        self.add_message("user", user_input)
        
        # Update status
        self.update_status("Thinking...")
        
        # Process in a separate thread to avoid freezing UI
        threading.Thread(target=self._process_in_thread, args=(user_input,), daemon=True).start()
        
    def _process_in_thread(self, user_input):
        """Process the user input in a separate thread"""
        # Get AI response
        response = self.therapist.chat_with_ai(user_input)
        
        # Update GUI with response
        self.root.after(0, lambda: self.add_message("assistant", response))
        self.root.after(0, lambda: self.update_status("Ready"))
        
        # Speak the response if voice output is enabled
        if self.voice_output_var.get():
            self.root.after(100, lambda: self.therapist.speak_text(response))
        
    def add_message(self, role, content):
        """Add a message to the conversation display"""
        self.conversation_text.config(state=tk.NORMAL)
        
        if role == "user":
            self.conversation_text.insert(tk.END, "You: ", "user_tag")
            self.conversation_text.insert(tk.END, f"{content}\n\n", "user_message")
        else:
            self.conversation_text.insert(tk.END, "Therapist: ", "assistant_tag")
            self.conversation_text.insert(tk.END, f"{content}\n\n", "assistant_message")
            
        self.conversation_text.tag_config("user_tag", foreground="blue", font=("Arial", 11, "bold"))
        self.conversation_text.tag_config("user_message", font=("Arial", 11))
        self.conversation_text.tag_config("assistant_tag", foreground="green", font=("Arial", 11, "bold"))
        self.conversation_text.tag_config("assistant_message", font=("Arial", 11))
        
        self.conversation_text.config(state=tk.DISABLED)
        self.conversation_text.see(tk.END)
        
    def update_status(self, status):
        """Update the status label"""
        self.status_label.config(text=status)
        
    def clear_conversation(self):
        """Clear the conversation display"""
        self.conversation_text.config(state=tk.NORMAL)
        self.conversation_text.delete(1.0, tk.END)
        self.conversation_text.config(state=tk.DISABLED)
        
        # Reset chat history
        self.therapist.chat_history = ChatMessageHistory()
        
        # Add welcome message
        self.add_message("assistant", "Chat cleared. How are you feeling now?")


def main():
    # Create the main window
    root = tk.Tk()
    root.configure(bg='white')
    
    # Set theme (if available)
    try:
        style = ttk.Style()
        style.theme_use('clam')  # Other options: 'alt', 'default', 'classic'
    except:
        pass
    
    # Create the application
    app = TherapistAIGUI(root)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()