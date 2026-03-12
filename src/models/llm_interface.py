# from transformers import pipeline


# def load_llm():
#     """
#     Load a text generation model.
#     """

#     generator = pipeline(
#         "text-generation",
#         model="google/flan-t5-base"
#     )

#     # generator = pipeline(
#     #     "text2text-generation",
#     #     model="google/flan-t5-base"
#     # )

#     return generator


# def generate_answer(query, context_chunks, generator):
#     """
#     Generate answer using retrieved document context.
#     """

#     context = "\n\n".join(context_chunks)

#     prompt = f"""
#     Answer the question using the context below.

#     Context:
#     {context}

#     Question:
#     {query}

#     Answer:
#     """

#     # response = generator(prompt, max_length=256)
#     response = generator(prompt, max_new_tokens=256)

#     answer = response[0]["generated_text"]

#     return answer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_llm():

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


def generate_answer(query, context_chunks, tokenizer, model):

    # context = "\n\n".join(context_chunks)
    context = "\n\n".join(context_chunks[:3])

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer