import json
import os
import re

import pandas as pd
import torch
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
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

# ---------------- Load Data ----------------
df = pd.read_excel("/path/to/data.xlsx")
df = df[["Index", "Comments", "Target", "Intent", "Implication"]].dropna()
test_df = df


# ---------------- Load Model ----------------
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


# ---------------- Prompt निर्माण ----------------
def format_prompt(comment, field):
    intro = """
You are an expert in analyzing hateful comments driven by fake narratives.

Based on the given comment, provide the requested analysis.
"""

    if field == "Target":
        return f"""{intro}
Identify the *Target* (who is being targeted) in the following comment.
If there is one target, only mention that. If there are multiple, mention each target as a single word and separate them by commas.

## Comment: "{comment}"
## Target:"""

    elif field == "Intent":
        return f"""{intro}
Briefly describe the *Intent* (motive or purpose behind the comment) using a single concise sentence.

## Comment: "{comment}"
## Intent:"""

    elif field == "Implication":
        return f"""{intro}
Briefly describe the possible *Implication* (impact or consequence on society) in a single concise sentence.

## Comment: "{comment}"
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
    if field == "Target":
        pattern = r'## Target:\s*(.+?)(?:\n## |\n\n|$)'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if not match:
            return ""
        text = match.group(1).strip()

        quoted = re.findall(r'"([^"]+)"', text)
        if quoted:
            return ", ".join(quoted).strip()

        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            if re.fullmatch(r'[\w\s,]+', line):
                line = re.sub(r'\s*,\s*', ', ', line)
                return line.strip(", ")

        return text

    elif field in ["Intent", "Implication"]:
        pattern = rf'## {field}:\s*(.+?)(?:\n## |\n\n|$)'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if not match:
            return ""

        text = match.group(1).strip()
        sentences = re.findall(r'[^.?!]*[.?!]', text)
        if not sentences:
            return text.splitlines()[0].strip()

        cleaned_text = " ".join(sentences[:2]).strip()
        return re.sub(r'\s+', ' ', cleaned_text)

    return ""


# ---------------- Embedding model ----------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------- Run Inference + Evaluation ----------------
generated_responses = []

similarities = {"Target": [], "Intent": [], "Implication": []}
bert_scores = {"Target": [], "Intent": [], "Implication": []}
rouge_scores = {"Target": [], "Intent": [], "Implication": []}
bleu_scores = {"Target": [], "Intent": [], "Implication": []}


for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    comment = row["Comments"]

    gen_data = {
        "Comment": comment,
        "Original Target": row["Target"],
        "Original Intent": row["Intent"],
        "Original Implication": row["Implication"]
    }

    for field in ["Target", "Intent", "Implication"]:
        prompt = format_prompt(comment, field)
        generated = generate_response(prompt)
        cleaned_generated = clean_response(generated, field)

        gen_data[f"Generated {field}"] = cleaned_generated

        # Sentence-BERT
        ref_emb = embedder.encode(str(row[field]), convert_to_tensor=True)
        gen_emb = embedder.encode(str(cleaned_generated), convert_to_tensor=True)
        sim_score = util.cos_sim(ref_emb, gen_emb).item()
        similarities[field].append(sim_score)
        gen_data[f"{field}_Similarity"] = sim_score

        # BERTScore
        _, _, F1 = bert_score([cleaned_generated], [str(row[field])],
                              lang="en", rescale_with_baseline=True)
        bert_scores[field].append(F1[0].item())
        gen_data[f"{field}_BERTScore"] = F1[0].item()

        # ROUGE
        rouge_result = rouge.score(str(row[field]), cleaned_generated)
        rouge_scores[field].append(rouge_result['rougeL'].fmeasure)
        gen_data[f"{field}_ROUGE"] = rouge_result['rougeL'].fmeasure

        # BLEU
        ref_tokens = str(row[field]).split()
        gen_tokens = cleaned_generated.split()
        bleu_score_val = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=bleu)
        bleu_scores[field].append(bleu_score_val)
        gen_data[f"{field}_BLEU"] = bleu_score_val

    generated_responses.append(gen_data)


# ---------------- Save Output ----------------
output_df = pd.DataFrame(generated_responses)
output_df.to_excel("/path/to/output.xlsx", index=False)


# ---------------- Print Averages ----------------
print("\n=== Average Evaluation Scores ===")
for field in ["Target", "Intent", "Implication"]:
    print(f"\n--- {field} ---")
    print(f"Sentence-BERT Similarity: {sum(similarities[field]) / len(similarities[field]):.4f}")
    print(f"BERTScore (F1): {sum(bert_scores[field]) / len(bert_scores[field]):.4f}")
    print(f"ROUGE-L F1: {sum(rouge_scores[field]) / len(rouge_scores[field]):.4f}")
    print(f"BLEU: {sum(bleu_scores[field]) / len(bleu_scores[field]):.4f}")
