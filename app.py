from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline
import time
import gradio as gr
import os
import speech_recognition as sr # Import the speech recognition library

# --- Part 1: AI Model Setup (Unchanged) ---

MODEL_DIR = "tinyllama_int8"
print(f"Loading model and tokenizer from: {MODEL_DIR}")

try:
    model = OVModelForCausalLM.from_pretrained(MODEL_DIR, device="CPU")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    cpu_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("Model and tokenizer loaded successfully and pipeline created for CPU.")
except Exception as e:
    print(f"--- FATAL ERROR: Could not load the AI model. ---")
    print(f"Please make sure the '{MODEL_DIR}' folder exists and was created correctly.")
    input("\nPress Enter to exit.")
    exit()


# --- Part 2: Backend Functions ---

def listen_and_transcribe():
    """
    Uses the SpeechRecognition library to listen to the microphone and return text.
    This function is now the backend for our "Listen" button.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening...")
        
        try:
            # Listen to the microphone input
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Recognizing speech...")
            # Use Google's online service to transcribe the audio
            text = recognizer.recognize_google(audio)
            print(f"Recognized text: {text}")
            # Return the transcribed text to be placed in the UI textbox
            return text
        except sr.WaitTimeoutError:
            return "Error: No speech detected."
        except sr.UnknownValueError:
            return "Error: Could not understand audio."
        except sr.RequestError as e:
            return f"Error: Service request failed. Check internet."
        except Exception as e:
            return f"Error: An unexpected error occurred: {e}"


def run_studybuddy(task_type, user_input):
    """
    The main AI function, processes text input and returns a response.
    """
    if not user_input or not user_input.strip():
        return "Please type a question or text to summarize in the box."

    print(f"Received task: {task_type}, Input: '{user_input[:30]}...'")
    
    if task_type == "Ask a Question":
        prompt = user_input
    else: # Summarize Text
        prompt = f"Summarize the following text in three clear bullet points:\n\n{user_input}"

    chat_prompt = f"<|system|>\nYou are a helpful study assistant. Your answers should be concise and clear.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    
    print("Generating response on CPU...")
    start_time = time.time()
    outputs = cpu_pipeline(chat_prompt, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9)
    end_time = time.time()
    latency = end_time - start_time
    
    response_text = outputs[0]['generated_text']
    assistant_response = response_text.split("<|assistant|>")[-1].strip()

    final_output = f"{assistant_response}\n\n--- \n*Generated on CPU in {latency:.2f} seconds.*"
    return final_output


# --- Part 3: Create and Launch the Gradio Web Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="StudyBuddy AI") as demo:
    gr.Markdown("# StudyBuddy AI\n### Powered by OpenVINO on Intel CPU")
    
    with gr.Row():
        with gr.Column(scale=2):
            task_type = gr.Radio(
                ["Ask a Question", "Summarize Text"], 
                label="Choose a Task", 
                value="Ask a Question"
            )
            
            # This is the textbox where the user types OR where transcribed text appears
            user_input = gr.Textbox(
                lines=8,
                label="Your Input", 
                placeholder="Type your question or click 'Listen' to use your voice."
            )
            
            # The buttons that trigger the actions
            with gr.Row():
                listen_button = gr.Button("🎙️ Listen")
                submit_button = gr.Button("Get Answer", variant="primary")

        with gr.Column(scale=3):
            output_text = gr.Textbox(
                lines=15, 
                label="AI Assistant's Response", 
                interactive=False
            )

    # --- Wire up the buttons to the backend functions ---

    # 1. When the "Listen" button is clicked...
    listen_button.click(
        fn=listen_and_transcribe, # ...call our speech-to-text function...
        inputs=None,              # ...it takes no inputs...
        outputs=user_input        # ...and its output (the text) goes into the user_input textbox.
    )

    # 2. When the "Get Answer" button is clicked...
    submit_button.click(
        fn=run_studybuddy, # ...call our main AI function...
        inputs=[task_type, user_input], # ...it takes the task type and the text from the input box...
        outputs=output_text # ...and its output goes into the main output textbox.
    )

print("Launching Gradio UI... Open the URL in your browser.")
demo.launch(share=True)