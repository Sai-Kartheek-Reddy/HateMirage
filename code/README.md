# HateMirage

## Prompt Template for Explanation Generation

```text

You are an expert in analyzing hateful comments driven by fake narratives.

Based on the given comment, provide the requested analysis.

---

### 1. Target
Identify the Target (who is being targeted) in the following comment.
- If there is one target, only mention that.
- If there are multiple targets, mention each target as a single word, separated by commas.
- Respond only with the target(s); do not provide any explanation.

## Comment: "{comment}"
## Target:

---

### 2. Intent
Briefly describe the Intent (motive or purpose behind the comment) using a single concise sentence.

## Comment: "{comment}"
## Intent:

---

### 3. Implication
Briefly describe the possible Implication (impact or consequence on society) in a single concise sentence.

## Comment: "{comment}"
## Implication:

---

```

**Note**: When using RAG, additional context will be incorporated into the prompt to provide grounded explanations.

## RAG Configuration

The Retrieval-Augmented Generation (RAG) setup is used to provide grounded contextual information during explanation generation.

**RAG Setup Summary:**
- **Embedding model:** `sentence-transformers/all-mpnet-base-v2`
- **Retriever backend:** FAISS (dense vector similarity search)
- **Top-k documents retrieved:** 5
- **Similarity metric:** Cosine similarity
- **Context source:** Data collected from credible web articles related to fake claims

The retrieved context is concatenated with the input comment to help the model generate grounded and context-aware explanations, reducing hallucination and improving factual consistency.

