from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.models.prompt_templates import build_rag_prompt

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_llm():

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    return tokenizer, model


def generate_answer(query, context_chunks, tokenizer, model, max_tokens = 300):

    context = "\n\n".join(context_chunks[:3])

#     prompt = f"""
# You are a helpful AI research assistant.

# Answer clearly and directly in 1-3 sentences.

# Context:
# {context}

# Question:
# {query}

# Answer:
# """

    prompt = f"""
You are a helpful AI research assistant.

Answer the question using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.3,
        do_sample=True,
        top_p = 0.9
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean output
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    return answer