# conda activate unsloth_env

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from datasets import load_dataset
import re
import os
import datetime
import wandb

max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/tinyllama", # "unsloth/tinyllama-bnb-4bit" for 16bit loading
    model_name = "unsloth/tinyllama-bnb-4bit", #for 16bit loading
    # model_name = "unsloth/Qwen2-0.5b-bnb-4bit", #for 16bit loading
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # return_dict=True
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 2,#32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 4,#32,
    lora_dropout = 0, # Currently only supports dropout = 0
    bias = "none",    # Currently only supports bias = "none"
    use_gradient_checkpointing = True, # @@@ IF YOU GET OUT OF MEMORY - set to True @@@
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

## DATASET
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
  full_text = []
  if not isinstance(examples['prompt'], list):
    # print("Non lista:", len(examples['prompt']))
    full_text = [f"{examples['prompt']} {examples['completion'].strip()}{EOS_TOKEN}"]
  else:
    # print(len(examples['prompt']))
    for i in range(len(examples['prompt'])):
      full_text.append(f"{examples['prompt'][i]} {examples['completion'][i].strip()}{EOS_TOKEN}")
  return full_text

dataset = load_dataset("Paoloc99/dataset", split="train[:70000]")
#dataset = dataset.map(formatting_prompts_func, batched = True,)
eval_dataset = load_dataset("Paoloc99/dataset", split="train[-1000:]")
# print(len(test_dataset))

response_template_with_context = "\nAnswer:"  # We added context here: "\n". This is enough for this tokenizer
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[22550, 29901]`
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

## CHECKPOINT
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d")

checkpoint_dir = 'checkpoints'
output_dir = f'checkpoints/{timestamp}'

def get_latest_checkpoint_dir(base_dir):
    # Ottieni tutte le sottocartelle con formato di timestamp
    timestamp_dirs = [d for d in os.listdir(base_dir) if re.match(r'\d{4}-\d{2}-\d{2}', d)]

    # Ordina le directory per timestamp
    timestamp_dirs.sort(key=lambda date: datetime.datetime.strptime(date, "%Y-%m-%d"), reverse=True)

    # Prendi la directory con il timestamp più recente
    latest_timestamp_dir = timestamp_dirs[0] if timestamp_dirs else None
    return os.path.join(base_dir, latest_timestamp_dir) if latest_timestamp_dir else None

latest_timestamp_dir = get_latest_checkpoint_dir(checkpoint_dir)

def get_latest_checkpoint(checkpoint_dir):
    if checkpoint_dir is None:
        return None

    # Ottieni tutte le sottocartelle con formato "checkpoint-{numero}"
    checkpoint_dirs = [d for d in os.listdir(checkpoint_dir) if re.match(r'checkpoint-\d+', d)]

    # Estrai i numeri dai nomi delle cartelle e ordina in base al numero
    checkpoint_dirs.sort(key=lambda x: int(re.search(r'\d+', x).group()), reverse=True)

    # Prendi la directory con il numero più grande
    latest_checkpoint_dir = checkpoint_dirs[0] if checkpoint_dirs else None
    return os.path.join(checkpoint_dir, latest_checkpoint_dir) if latest_checkpoint_dir else None

checkpoint_path = get_latest_checkpoint(latest_timestamp_dir)
# print(checkpoint_path)
if checkpoint_path and not os.path.exists(checkpoint_path):
    print(f"Checkpoint {checkpoint_path} does not exist.")
else:
    print(f"Checkpoint {checkpoint_path} found.")

## TRAINER
set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="lost-in-the-middle"
# set the wandb project where this run will be logged
os.environ["WANDB_NOTEBOOK_NAME "]="lost-in-the-middle-nb"
# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="checkpoint"
# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"
wandb.login(key="372f5c298afc4be9b40dd7b97523d394c3d30d05")
wandb.init(project="lost-in-the-middle", name="test_loss")

os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import numpy as np

def compute_metrics(eval_pred):

    label_ids = eval_pred.label_ids
    # Rimuove i token speciali dai label_ids prima della decodifica
    label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
    label_texts = [tokenizer.decode(label_id, skip_special_tokens=True) for label_id in label_ids]

    logits = eval_pred.predictions[0]
    logits_ids = np.where(logits == -100, tokenizer.pad_token_id, logits)
    pred_texts = [tokenizer.decode(logit_id, skip_special_tokens=True) for logit_id in logits_ids]

    em_correct = 0
    qem_correct = 0
    total = len(label_texts)

    # The-Power-of-Noise, exact match
    for label, pred in zip(label_texts, pred_texts):
        em_correct += 1 if label == pred else 0

    em = em_correct / total if total > 0 else 0
    
    # quasi-exact match
    for label, pred in zip(label_texts, pred_texts):
        qem_correct += 1 if label in pred else 0

    qem = qem_correct / total if total > 0 else 0

    # partial match
    from evaluate import load
    exact_match_metric = load("exact_match")
    results = exact_match_metric.compute(predictions=pred_texts, references=label_texts)
    exact_match = results["exact_match"]

    # Perplexity e^loss
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=pred_texts, model_id='unsloth/tinyllama')
    mean_perplexity = round(results["mean_perplexity"], 2)
    perplexities = round(results["perplexities"][0], 2)

    # shift_logits = torch.tensor(logits_ids[..., :-1, :]).contiguous()
    # shift_labels = torch.tensor(label_ids[..., 1:]).contiguous()
    logits_tensor = torch.tensor(eval_pred.predictions[1])
    labels_tensor = torch.tensor(eval_pred.label_ids)
    with torch.no_grad():
        loss = torch.nn.CrossEntropyLoss()(logits_tensor.view(-1, logits_tensor.size(-1)), labels_tensor.view(-1))
    
    perplexity = torch.exp(loss).item()

    print('-----')
    # print(em)
    # print('-----')
    return {"em" : em, "qem" : qem, "exact_match" : exact_match, "perplexity" : perplexity }

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    print(logits)
    print(labels)
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = eval_dataset,
    max_seq_length = max_seq_length,
    # dataset_num_proc = 2,
    packing = False, # Packs short sequences together to save time!
    formatting_func=formatting_prompts_func,
    data_collator=collator,

    # compute_metrics= compute_metrics, # Funzione che restituisce dizionario con una chiave per ogni metrica                    
    # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    args = TrainingArguments(
        # report_to="wandb",
        # run_name= "test_eval",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        # Se steps < 4000 -> 0.1, se 20.000 ->
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        learning_rate = 2e-5,
        # learning_rate = 1e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim = "adamw_8bit",
        weight_decay = 0.1,
        lr_scheduler_type = "linear",#"cosine",
        seed = 3407,
        output_dir = output_dir,
        logging_steps=5,
        log_level='debug', #'info',
        save_steps=100,
        save_total_limit=10,
        save_strategy="steps",
        eval_steps=100,
        eval_strategy="steps",
        do_eval = True,
        load_best_model_at_end = True,
        # metric_for_best_model= Già settato alla loss
    ),
)

###################################
# def preprocess_function(examples):
#     # Tokenizza i prompt e le completions
#     prompts = examples['prompt']
#     completions = examples['completion']
#     full_texts = formatting_prompts_func(examples)
    
#     # Tokenizza i testi completi
#     tokenized_inputs = tokenizer(full_texts, truncation=True, padding=True, max_length=max_seq_length)
#     return tokenized_inputs

# tokenized_eval_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)
# predictions = trainer.predict(tokenized_eval_dataset)
# print(predictions.predictions[0].shape, predictions.predictions[1].shape, predictions.label_ids.shape)

###################################

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# data_collator = trainer.get_train_dataloader().collate_fn
# actual_train_set = trainer._remove_unused_columns(trainer.train_dataset)
# batch = data_collator([actual_train_set[i] for i in range(4)])
# print("Il Trainer setta EOS TOKEN: ", not False in [EOS_TOKEN in tokenizer.decode(batch['input_ids'][i]) for i in range(len(batch['input_ids']))] )

## TRAINING
trainer_stats = trainer.train(resume_from_checkpoint=checkpoint_path)

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

## INFERENCE
index = 1
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    eval_dataset[index]["prompt"]
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, use_cache = True)
print(tokenizer.batch_decode(outputs)[0])
print("-------------")
print("Risposta esatta: ", eval_dataset[index]["completion"])

## SAVE
model.save_pretrained(checkpoint_dir + "/litm_model") # Local saving
tokenizer.save_pretrained(checkpoint_dir + "/litm_model")
model.push_to_hub("Paoloc99/litm_model", token="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") # Online saving
tokenizer.push_to_hub("Paoloc99/litm_model", token="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") # Online saving