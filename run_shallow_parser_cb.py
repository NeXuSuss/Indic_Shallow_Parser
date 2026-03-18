import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from shallow_parser import shallow_parse

# === CB Model Functions ===
def load_cb_model(model_path, hf_token):
    config = PeftConfig.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="cuda", 
        torch_dtype=torch.bfloat16,
        token=hf_token
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_cb_annotation(model, tokenizer, raw_sentence):
    instruction = """Annotate the following Hindi sentence with clause boundaries.

CLAUSE TYPES:
- MCL (Main Clause)
- RCL (Relative Clause) 
- RP (Relative Participle)
- COND (Conditional)
- NF (Non-Finite)
- INF (Infinitive)
- COM (Complementizer)
- ADVCL (Adverbial)

FORMAT: TYPE[ text ]TYPE (nesting allowed)

Input: {input}
Output:"""
    prompt = instruction.format(input=raw_sentence)
    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    prediction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return prediction.strip()

def parse_cb_to_boundaries(cb_output, tokens):
    boundary_tags = [[] for _ in tokens]
    pattern = r'(\w+)\[\s*([^\[\]]+)\s*\]\1'
    temp_output = cb_output
    all_clauses = []
    
    while True:
        match = re.search(pattern, temp_output)
        if not match:
            break
        clause_type = match.group(1)
        clause_text = match.group(2).strip()
        clause_tokens = clause_text.split()
        all_clauses.append((clause_type, clause_tokens))
        temp_output = temp_output[:match.start()] + clause_text + temp_output[match.end():]
    
    all_clauses.reverse()
    for clause_type, clause_tokens in all_clauses:
        ct_idx = 0
        start_pos = -1
        end_pos = -1
        for i, token in enumerate(tokens):
            if ct_idx < len(clause_tokens) and token == clause_tokens[ct_idx]:
                if ct_idx == 0: start_pos = i
                if ct_idx == len(clause_tokens) - 1:
                    end_pos = i
                    break 
                ct_idx += 1
        if start_pos != -1: boundary_tags[start_pos].append(f"{{{clause_type}}}")
        if end_pos != -1: boundary_tags[end_pos].append(f"{{/{clause_type}}}")
    return boundary_tags

def parse_morph(morph_str):
    feats = {"Gender": "", "Number": "", "Person": "", "Case": "", "Vib": ""}
    for item in morph_str.split("|"):
        if "=" in item:
            k, v = item.split("=", 1)
            if k in feats: feats[k] = v
    return feats

def ssf_string_to_fs_with_cb(ssf_text, cb_tags, fileno=1, starting_sentno=0):
    output_lines = []
    idx = 0
    sentno = starting_sentno
    token_id = 0 

    for line in ssf_text:
        line = line.strip()
        if not line:
            if token_id > 0: 
                sentno += 1
                token_id = 0
            continue
            
        cols = line.split("\t")
        if len(cols) < 8: continue
            
        token, lemma, lcat, pos, morph, chunk = cols[1], cols[2], cols[3], cols[4], cols[5], cols[7]
        feats = parse_morph(morph)
        fs = f"{lemma},{lcat},{feats['Gender']},{feats['Number']},{feats['Person']},{feats['Case']},{feats['Vib']}"
        
        ner_tag = "O" 
        cb_tag_str = "".join(cb_tags[idx]) if (idx < len(cb_tags) and cb_tags[idx]) else "O"
            
        out_line = f"{fileno}\t{sentno}\t{token_id}\t{token}\t{pos}\t{chunk}\t{fs}\t{ner_tag}\t{cb_tag_str}"
        output_lines.append(out_line)
        idx += 1
        token_id += 1
        
    return "\n".join(output_lines)

# === Config (Updated for GitHub) ===
HF_TOKEN = "YOUR_TOKEN_HERE"
# Assumes you have a folder named 'model' inside your repo
CB_MODEL_PATH = "./model" 
# Assumes your input file is in a folder named 'tests'
INPUT_FILE_PATH = "./tests/input" 
# Output will be saved in the tests folder
OUTPUT_FILE_PATH = "./tests/output.txt" 

# === Main Pipeline ===
def run_pipeline():
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

    if not os.path.exists(INPUT_FILE_PATH):
        print(f"Error: {INPUT_FILE_PATH} not found.")
        return

    print("Loading CB model...")
    cb_model, cb_tokenizer = load_cb_model(CB_MODEL_PATH, HF_TOKEN)
    print("Models ready. Starting processing...")

    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f_out:
        
        current_sentno = 0
        for line in f_in:
            clean_text = line.strip()
            if not clean_text: continue
            
            # Clean leading space or source markers 
            clean_text = re.sub(r'^\\s*', '', clean_text).strip()
            
            conll_output = shallow_parse(clean_text, "hin", mode="conll")
            
            text_for_cb = clean_text.replace(" ।", "").replace("।", "").strip()
            cb_output = get_cb_annotation(cb_model, cb_tokenizer, text_for_cb)
            
            tokens = [l.split("\t")[1] for l in conll_output if l.strip() and len(l.split("\t")) >= 2]
            cb_boundary_tags = parse_cb_to_boundaries(cb_output, tokens)
            
            final_ssf = ssf_string_to_fs_with_cb(conll_output, cb_boundary_tags, starting_sentno=current_sentno)
            
            f_out.write(final_ssf + "\n\n")
            print(f"Processed sentence {current_sentno}")
            current_sentno += 1

    print(f"Done! Results saved to {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    run_pipeline()