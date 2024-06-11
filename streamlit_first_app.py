import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

with st.sidebar:
    #with st.echo():
    st.write("This code will be printed to the sidebar.")

st.title("💬 Chatbot")

# Initial welcome message
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
model_name = "apple/OpenELM-270M"
tokenizer_name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
#pipe = pipeline("text-generation", model=model_name, trust_remote_code=True, tokenizer=tokenizer)
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    output = model.generate(input_ids, max_length=100, pad_token_id=0,repetition_penalty=2.0, num_return_sequences=1)

    # Decode the generated text
    completion = tokenizer.decode(output[0], skip_special_tokens=True)

    #completion = pipe(prompt,max_new_tokens=1028, num_return_sequences=1)
    print(completion)
    st.session_state.messages.append({"role": "assistant", "content": completion})
    st.chat_message("assistant").write(completion)
