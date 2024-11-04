import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import re
import os
import json
# from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
import evaluate
from bert_score import score

def save_metrics(metrics, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename + ".json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def generate_batch_texts(prompts, model, tokenizer, max_length=4096):
    
    # Tokenizzazione dei prompt
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to('cuda')
    
    # Generazione del testo
    outputs = model.generate(**inputs, max_new_tokens=10, use_cache=False)
    
    # Decodifica del testo generato
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    # Rimozione del testo prima di "Answer: " in ogni prompt
    modified_generated_output = []
    for text in generated_texts:
        modified_output = text.split("Answer:", 1)[1]
        modified_output = remove_punctuation(modified_output.lower()).strip()
        modified_generated_output.append(modified_output)

    answers_ids = tokenizer(modified_generated_output, return_tensors="pt", padding=True, truncation=True).to('cuda')
    answers_ids = answers_ids['input_ids']

    # Rimozione degli elementi 0 e 1 da ciascun tensore nella lista
    cleaned_input_ids = []
    for tensor in answers_ids:
        # Filtra il tensore per mantenere solo gli ID diversi da 0 e 1
        filtered_tensor = tensor[tensor > 1]
        cleaned_input_ids.append(filtered_tensor)

    return cleaned_input_ids, modified_generated_output

def generate_true_ids(prompts, model, tokenizer, max_length=4096):
    # Tokenizzazione dei prompt
    prompts = [remove_punctuation(prompt.lower()) for prompt in prompts]
    outputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to('cuda')
    outputs = outputs['input_ids']
    
    # Rimozione degli elementi 0 e 1 da ciascun tensore nella lista
    cleaned_input_ids = []
    for tensor in outputs:
        # Filtra il tensore per mantenere solo gli ID diversi da 0 e 1
        filtered_tensor = tensor[tensor > 1]
        cleaned_input_ids.append(filtered_tensor)
    
    return cleaned_input_ids

def match_tokens(generated_texts_list, true_texts_list):
    tp = 0  # Veri positivi
    fp = 0  # Falsi positivi
    fn = 0  # Falsi negativi

    # Itera su ogni coppia di tensori nella lista
    for true_list, generated_list in zip(true_texts_list, generated_texts_list):        
        # Conta la frequenza di ciascun token nelle liste
        true_counter = Counter(true_list)
        generated_counter = Counter(generated_list)
        
        # Calcola i veri positivi e i falsi negativi
        for token in true_counter:
            if token in generated_counter:
                tp += min(true_counter[token], generated_counter[token])
                fn += max(0, true_counter[token] - generated_counter[token])
            else:
                fn += true_counter[token]
        
        # Calcola i falsi positivi
        for token in generated_counter:
            if token not in true_counter:
                fp += generated_counter[token]
            elif generated_counter[token] > true_counter[token]:
                fp += generated_counter[token] - true_counter[token]
    
    return tp, fp, fn

def compute_partial_match(generated_texts, true_texts):
    pm = 0

    # Itera su ogni coppia di tensori nella lista
    for true_list, generated_list in zip(generated_texts, true_texts):        
        # Conta la frequenza di ciascun token nelle liste
        true_counter = Counter(true_list)
        generated_counter = Counter(generated_list)

        # Calcola i falsi positivi
        for token in generated_counter:
            if token in true_counter:
                pm += 1
                break
    
    return pm/len(true_texts)

def compute_partial_match_jaccard(generated_texts, true_texts):
    pmj = 0

    # Itera su ogni coppia di tensori nella lista
    for true_list, generated_list in zip(true_texts, generated_texts):     
        true_set = set(true_list)   
        generated_set = set(generated_list)
        
        pmj += len(true_set.intersection(generated_set)) / len(true_set.union(generated_set))
        
    return pmj/len(true_texts)

def compute_exact_match(generated_texts, true_texts):
    em = sum([1 if p == l else 0 for p, l in zip(generated_texts, true_texts)]) / len(true_texts)
    return em

def compute_rouge(generated_texts, true_texts):
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=generated_texts,
                            references=true_texts,
                            use_aggregator=True)
    return results

def compute_bleu(generated_texts, true_texts):
    bleu = evaluate.load("bleu")
    true_list = [[text] for text in true_texts]
    results = bleu.compute(predictions=generated_texts, references=true_list)
    return results['bleu']

def compute_bert_score(generated_texts, true_texts):
    P, R, F1 = score(generated_texts, true_texts, lang='en', verbose=True)
    return P.mean().item(), R.mean().item(), F1.mean().item()

def compute_metrics(generated_texts, true_texts):
    # Calculate Exact Match (EM)
    em = compute_exact_match(generated_texts, true_texts) 

    # Calculate Partial Match (PM)
    pm = compute_partial_match(generated_texts, true_texts)

    # Calculate Partial Match Jaccard (PMJ)
    pmj = compute_partial_match_jaccard(generated_texts, true_texts)

    # Calculate F1
    tp, fp, fn = match_tokens(generated_texts, true_texts)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Calculate ROUGE
    rouge_dict = compute_rouge(generated_texts, true_texts)

    # Calculate BLEU
    bleu = compute_bleu(generated_texts, true_texts)

    # Compute Precision, Recall and F1 via BERT Score
    bert_precision, bert_recall, bert_f1 = compute_bert_score(generated_texts, true_texts)

    return {'exact_match' : em, 'partial_match' : pm, 'jaccard_partial_match' : pmj, 
            'precision' : precision, 'recall' : recall, 'f1' : f1,
            'rouge1' : rouge_dict['rouge1'], 'rouge2' : rouge_dict['rouge2'], 'rougeL' :rouge_dict['rougeL'],
            'rougeLsum': rouge_dict['rougeLsum'], 'bleu' : bleu,
            'bert_precision' : bert_precision, 'bert_recall' : bert_recall, 'bert_f1' : bert_f1}

def main(args):
    
    max_seq_length = args.max_seq_length
    dtype = torch.float16 if args.dtype == 'float16' else torch.bfloat16 if args.dtype == 'bfloat16' else None
    load_in_4bit = args.load_in_4bit
    use_liger = args.use_liger
    # gradient_checkpointing = args.gradient_checkpointing
    use_reentrant = True
    output_dir = args.output_dir
    filename = args.filename
    model_name = args.model_name

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4"
    )

    if use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model = AutoLigerKernelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype,
            quantization_config=bnb_config,
            # use_cache= not gradient_checkpointing,
            use_reentrant=use_reentrant
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.gradient_checkpointing_disable()
    else:           
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            trust_remote_code=True
        )
        FastLanguageModel.for_inference(model)

    # eval_dataset = load_dataset("Paoloc99/dataset", split="test[:100]")
    eval_dataset = load_dataset("Paoloc99/dataset", split="test[:1000]")
    
    # Definire la lunghezza massima consentita
    max_allowed_length = max_seq_length

    # Funzione per calcolare la lunghezza di ciascun esempio
    def filter_long_inputs(example):
        tokenized_input = tokenizer(example['prompt'], truncation=False)
        input_length = len(tokenized_input['input_ids'])
        return input_length <= max_allowed_length  
    
    # Applicare il filtro al dataset
    eval_dataset = eval_dataset.filter(filter_long_inputs, batched=False)
    print("Lunghezza eval dataset dopo filter:", len(eval_dataset))

    dataloader_batch_size = 2
    data_loader = DataLoader(eval_dataset, batch_size=dataloader_batch_size)
    prompts, generated_texts, generated_ids, true_texts, true_ids = [], [], [], [], []
    # Loop attraverso i batch
    for batch in tqdm(data_loader):
        prompts.extend(batch['prompt'])  # Lista di prompt nel batch
        true_texts.extend([remove_punctuation(el.lower()) for el in batch['completion']])
        
        # Genera il testo per l'intero batch
        batch_generated_ids, batch_generated_texts = generate_batch_texts(batch['prompt'], model, tokenizer)
        # batch_true_ids = generate_true_ids(batch['completion'], model, tokenizer)

        # Salva i testi generati
        generated_texts.extend(batch_generated_texts)
        # generated_ids.extend(batch_generated_ids)
        # true_ids.extend(batch_true_ids)
    metrics = compute_metrics(generated_texts, true_texts)
    save_metrics(metrics, output_dir, filename)
    print(f"Metrics saved to {output_dir}/{filename}.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the model and save metrics")
    parser.add_argument("--model_name", type=str, required=True, help="The name or path of the model to use")
    parser.add_argument("--output_dir", type=str, default="results_metrics", help="Directory to save the metrics")
    parser.add_argument("--filename", type=str, default="metrics", help="Filename to save the metrics")
    parser.add_argument('--use_liger', type=bool, default=False, help='Whether to use LIGER kernel.')
    parser.add_argument('--load_in_4bit', type=bool, default=True, help='Whether to load model in 4-bit.')
    parser.add_argument('--max_seq_length', type=int, default=4096, help='Maximum sequence length.')
    parser.add_argument('--dtype', type=str, default='None', choices=['float16', 'bfloat16', 'None'], help='Data type to use.')
    args = parser.parse_args()
    main(args)