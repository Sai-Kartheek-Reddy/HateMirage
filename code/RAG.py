import json
import os
import re

import pandas as pd
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

# ---------------- Evaluation helpers ----------------
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
bleu = SmoothingFunction().method4()

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Load Dataset ----------------
df = pd.read_excel("/path/to/data.xlsx")
df = df[["Index", "Comments", "Target", "Intent", "Implication"]].dropna()
test_df = df


# ---------------- Load RAG Documents ----------------
jsonl_path = "/path/to/rag_docs.jsonl"

documents = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        documents.append(
            Document(
                page_content=item["text"],
                metadata={
                    "topic": item.get("topic"),
                    "chunk_id": item.get("chunk_id"),
                    "tags": item.get("tags", [])
                }
            )
        )


# ---------------- Build / Load FAISS ----------------
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

faiss_db = FAISS.from_documents(documents, embeddings)
faiss_db.save_local("/path/to/faiss_index")

faiss_db = FAISS.load_local(
    "/path/to/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# ---------------- Load LLM ----------------
model_id = "model-id" # Model-id should match with the huggingface-id of the model you want to use. For example, "gpt2", "EleutherAI/gpt-j-6B", etc.

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config
)


# ---------------- Retrieval ----------------
def search_rag_context(query, top_k=5):
    results = faiss_db.similarity_search(query, k=top_k)
    return " ".join([doc.page_content for doc in results])


# ---------------- Prompt ----------------
def format_prompt_with_context(comment, context, field):
    intro = f"""
You are an expert in analyzing hateful comments driven by fake narratives.

The following context provides background information retrieved from external sources:
[Context]: "{context}"

Based on the context and the comment, provide the requested analysis.
"""

    if field == "Target":
        return f"""{intro}
## Comment: "{comment}"
## Task: If there is one target, only mention that. If there are multiple, mention each target as a single word and separate them by commas.
## Target:"""

    elif field == "Intent":
        return f"""{intro}
## Comment: "{comment}"
## Task: Briefly describe the *Intent* in a single concise sentence.
## Intent:"""

    elif field == "Implication":
        return f"""{intro}
## Comment: "{comment}"
## Task: Briefly describe the possible *Implication* in a single concise sentence.
## Implication:"""


# ---------------- Generation ----------------
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


# ---------------- Cleaning ----------------
def clean_response(response, field):
    pattern = rf'## {field}:\s*(.+?)(?:\n## |\n\n|$)'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


# ---------------- Similarity Model ----------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------- Run RAG Inference ----------------
generated_responses = []
similarities = {"Target": [], "Intent": [], "Implication": []}

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    comment = row["Comments"]

    gen_data = {
        "Comment": comment,
        "Original Target": row["Target"],
        "Original Intent": row["Intent"],
        "Original Implication": row["Implication"]
    }

    # Retrieve context (no summarization)
    context = search_rag_context(comment, top_k=5)
    gen_data["Context"] = context

    for field in ["Target", "Intent", "Implication"]:
        prompt = format_prompt_with_context(comment, context, field)
        generated = generate_response(prompt)
        cleaned_generated = clean_response(generated, field)
        gen_data[f"Generated {field}"] = cleaned_generated

        ref_emb = embedder.encode(row[field], convert_to_tensor=True)
        gen_emb = embedder.encode(cleaned_generated, convert_to_tensor=True)
        score = util.cos_sim(ref_emb, gen_emb).item()
        similarities[field].append(score)
        gen_data[f"{field}_Similarity"] = score

    generated_responses.append(gen_data)


# ---------------- Save Output ----------------
output_df = pd.DataFrame(generated_responses)
output_df.to_excel("/path/to/output.xlsx", index=False)


# ---------------- Print Averages ----------------
print("\n=== Average Similarity Scores ===")
for field in similarities:
    avg_score = sum(similarities[field]) / len(similarities[field])
    print(f"{field}: {avg_score:.4f}")
