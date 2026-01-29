import torch
import json
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURATION ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATA_FILE = os.path.join(os.path.dirname(__file__), "intent_data.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_prompt(sample):
    """
    Constructs a multiple-choice prompt.
    """
    prompt = (
        "Classify the following user query into one of these categories:\n"
        "0 - Weather\n"
        "1 - Calendar\n"
        "2 - Chat\n\n"
        f"Query: {sample['question']}\n"
        "Answer:"
    )
    return prompt


def score_label_sequence(logits, token_seq):
    """
    Computes summed log-probability of a multi-token label sequence.
    This is required because Phi-3 digits are NOT single tokens.
    """
    score = 0.0
    log_probs = torch.log_softmax(logits, dim=-1)

    # We score tokens relative to the end of the prompt
    for i, token_id in enumerate(token_seq):
        score += log_probs[0, -len(token_seq) + i, token_id]

    return score


def evaluate():
    print(f"Loading {MODEL_NAME} on {DEVICE}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=False,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()

    # Load data
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    # Build full token sequences for labels (IMPORTANT FIX)
    label_token_seqs = {
        i: tokenizer.encode(str(i), add_special_tokens=False)
        for i in range(3)
    }

    logits_score = 0
    parsing_score = 0
    total = len(data)

    print(f"\nStarting evaluation on {total} samples...\n")

    for sample in data:
        prompt = build_prompt(sample)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # -------------------------------
        # METRIC 1: LOGITS (sequence-aware)
        # -------------------------------
        with torch.no_grad():
            outputs = model(**inputs)

        label_scores = {}
        for label, token_seq in label_token_seqs.items():
            label_scores[label] = score_label_sequence(outputs.logits, token_seq)

        predicted_idx = max(label_scores, key=label_scores.get)

        if predicted_idx == sample["answer"]:
            logits_score += 1
        else:
            print("Mismatch (logits)!")
            print(f"  Query: {sample['question']}")
            print(f"  Expected: {sample['answer']}")
            print(f"  Predicted: {predicted_idx}")

        # -------------------------------
        # METRIC 2: PARSING (generation)
        # -------------------------------
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False
            )

        output_text = tokenizer.decode(gen[0], skip_special_tokens=True)

        match = re.search(r"Answer:\s*([0-2])", output_text)
        if not match:
            match = re.search(r"\b([0-2])\b", output_text.split("Answer:")[-1])

        parsed_val = int(match.group(1)) if match else -1

        if parsed_val == sample["answer"]:
            parsing_score += 1

    # -------------------------------
    # FINAL REPORT
    # -------------------------------
    print("=" * 40)
    print(f"RESULTS for {MODEL_NAME}")
    print("=" * 40)
    print(f"Logits Accuracy:  {logits_score / total:.1%}")
    print(f"Parsing Accuracy: {parsing_score / total:.1%}")
    print("=" * 40)

    if logits_score > parsing_score:
        print("Analysis: Model knows the intent but struggles with output formatting.")
    elif parsing_score > logits_score:
        print("Analysis: Generation looks correct but internal confidence is weaker.")
    else:
        print("Analysis: Logits and Parsing are aligned.")


if __name__ == "__main__":
    print("Script started...")
    evaluate()