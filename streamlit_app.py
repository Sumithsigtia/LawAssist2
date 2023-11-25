import streamlit as st
import replicate
import os

# App title
st.set_page_config(page_title="Legal Assist Chatbot")
st.title('Ask me legal questions, and I\'ll provide answers!\n'
         'Created by: Vidhan Mehta, Sumith Sigtia, Shabiul Hasnain Siddiqui, Swathi')

# Replicate Credentials
with st.sidebar:
    st.title('Legal Assist Chatbot')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
            st.warning('Please enter your credentials!')
        else:
            st.success('Proceed to entering your prompt message!')

os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you with Indian legal matters today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you with Indian legal matters today?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
# Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    try:
        string_dialogue = "You are a legal assistant specializing in Indian law. You only respond once as 'Assistant'."
        for dict_message in st.session_state.messages:
            if dict_message["role"] == "user":
                string_dialogue += f"User: {dict_message['content']}\n\n"
            else:
                string_dialogue += f"Assistant: {dict_message['content']}\n\n"
        internal_prompt = "Assistant: I am acting as a legal assistant specializing in Indian law. Please provide details about your legal query or concern."
        input_prompt = f"{string_dialogue} {internal_prompt} {prompt_input} Assistant: "
        st.write(f"Input prompt: {input_prompt}")
        
        output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',
                               input={"prompt": input_prompt,
                                      "temperature": 0.1, "top_p": 0.9, "max_length": 512, "repetition_penalty": 1})
        st.write(f"Output from Replicate: {output}")
        
        return output

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api, placeholder="Please describe your Indian legal query or concern..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
