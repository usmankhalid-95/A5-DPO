# app.py
import os
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model and tokenizer from Hugging Face
pretrained_model_name = "UsmanTara/dpo_model" 
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)

# Check if an MPS device is available on Mac, or fallback to CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

# Streamlit UI setup
st.title("GPT-2 Text Generator")

# Input field for the user to provide a prompt
user_input = st.text_area("Please enter your prompt:")

# Function to structure the input prompt for the model
def create_prompt(user_input_text):
    return f"Human: {user_input_text}\n\nAssistant:"

# When the button is clicked, generate the model's response
if st.button("Generate Response"):
    if not user_input:
        st.error("No input provided! Please enter some text.")
    else:
        try:
            # Format the input prompt to match the expected structure
            structured_input = create_prompt(user_input)
        
            # Tokenize the formatted input text
            tokenized_input = tokenizer(structured_input, return_tensors="pt").to(device)
        
            # Generate text using the model
            with torch.no_grad():
                generated_output = model.generate(
                    tokenized_input.input_ids,
                    max_length=100,  # Adjust max length as needed
                    num_return_sequences=1,  # Generate one response at a time
                    temperature=0.7,  # Control randomness
                    top_p=0.92,  # Controls diversity of responses
                    repetition_penalty=1.2,  # Reduces repetitiveness
                    do_sample=True,  # Sampling strategy
                    pad_token_id=tokenizer.eos_token_id  # Padding for end-of-sequence token
                )
        
            # Decode the output and remove the prompt to get the response
            decoded_response = tokenizer.decode(generated_output[0], skip_special_tokens=True)
        
            # Extract only the model's response (exclude the prompt)
            assistant_response = decoded_response[len(structured_input):] if decoded_response.startswith(structured_input) else decoded_response
        
            # Display the generated response
            st.subheader("Assistant's Response:")
            st.write(assistant_response)

        except Exception as error:
            st.error(f"Something went wrong: {str(error)}")
