#LLM Functions & constants
from urllib import response
from functools import partial
from sae_functions import analyze_query_activations


def format(role, prompt):
  good_format = {"role": role, "content": prompt}
  return good_format
def tokenize_messages(model, tokenizer, messages):
  text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
  )
  tokenized_messages = tokenizer([text], return_tensors="pt").to(model.device)
  return tokenized_messages
def generate_normal_response(model, tokenizer,messages, num_tokens, temp):
  print("Generating response...\n")
  tokenized_messages = tokenize_messages(model, tokenizer, messages)
  generated_ids = model.generate(
      **tokenized_messages,
      max_new_tokens=num_tokens,   # Maximum number of tokens to generate
      temperature=temp,      # Controls randomness (lower = more deterministic)
      do_sample=True,       # Required if temperature is used
      pad_token_id=tokenizer.eos_token_id # Prevents padding warnings
  )
  generated_ids = [
      output_ids[len(input_ids):]
      for input_ids, output_ids in zip(tokenized_messages.input_ids, generated_ids)
  ]
  response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
  return response
def generate_sae_response(model,messages, num_tokens, temp):
  print("Generating response...\n")
  tokenizer = model.tokenizer
  text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
  input_ids = tokenizer.encode(text, return_tensors="pt").to(model.cfg.device)
  generated_ids = model.generate(
        input_ids,
        max_new_tokens=num_tokens,
        temperature=temp,
        verbose=False # Set to True if you want to see tokens stream in real-time!
    )
  new_tokens = generated_ids[0][input_ids.shape[1]:]
  response = tokenizer.decode(new_tokens, skip_special_tokens=True)
  return response
def chat_loop(model, tokenizer, SYSTEM_PROMPT, num_tokens, temp, is_sae, sae, top_k):
  history = []
  new_prompt = ""
  first_prompt = True
  while(True):
    new_prompt = input("Enter your message or 'Exit': ")
    if new_prompt.lower() == "exit": break
    analyze_query_activations(model, sae,new_prompt, top_k)
    if first_prompt: # If its the first prompt, append the System prompt too
        combined_prompt = f"{SYSTEM_PROMPT}\n\n{new_prompt}"
        history.append(format("user", combined_prompt))
        first_prompt = False
    else:
        history.append(format("user", new_prompt))
    response = generate_sae_response(model, tokenizer, history, num_tokens, temp) if (is_sae) else generate_normal_response(model, tokenizer, history, num_tokens, temp)
    print("LLM: ", response)
    history.append(format("assistant",response))




