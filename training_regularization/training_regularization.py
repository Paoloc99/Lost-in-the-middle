# conda activate unsloth_env
# from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, EarlyStoppingCallback, AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, Dataset
import re
import os
import datetime
from LITMTrainer import LITMTrainer
from peft import LoraConfig 
import random
from DataCollatorForLostInTheMiddle import DataCollatorForLostInTheMiddle
import wandb

max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
enable_litm = True

mu = 100
mu_str = str(mu).replace(".","_")

# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_name = "Qwen/Qwen2-0.5B-Instruct"
model_name = "unsloth/Qwen2-0.5B-Instruct-bnb-4bit"
# model_name = "unsloth/tinyllama-bnb-4bit"
# model_name = "Paoloc99/litm_model"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,device_map="auto", 
    # load_in_4bit=True, 
    # attn_implementation="eager", 
    torch_dtype = dtype,
    quantization_config=bnb_config,
    use_cache=False
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
peft_config = LoraConfig(
    lora_alpha= 4,
    lora_dropout= 0,
    r= 2,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
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
# dataset = load_dataset("Paoloc99/dataset", split="train[:1000]")
#dataset = dataset.map(formatting_prompts_func, batched = True,)
eval_dataset = load_dataset("Paoloc99/dataset", split="train[-1000:]")
# print(len(test_dataset))

response_template_with_context = "\nAnswer:"  # We added context here: "\n". This is enough for this tokenizer
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[-2:]  # Now we have it like in the dataset texts: `[22550, 29901]`
document_template_with_context = "\nDocument ["
document_template_ids = tokenizer.encode(document_template_with_context, add_special_tokens=False)[-2:]  # Now we have it like in the dataset texts: `[22550, 29901]`
question_template_with_context = "\nQuestion:"
question_template_ids = tokenizer.encode(question_template_with_context, add_special_tokens=False)[-2:]  # Now we have it like in the dataset texts: `[22550, 29901]`
collator = DataCollatorForLostInTheMiddle(response_template_ids, document_template=document_template_ids, question_template=question_template_ids, tokenizer=tokenizer)

## CHECKPOINT
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d")

checkpoint_dir = 'checkpoints-reg'
output_dir = f'checkpoints-reg/{timestamp}'

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
# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="lost-in-the-middle-reg"
# set the wandb project where this run will be logged
os.environ["WANDB_NOTEBOOK_NAME "]="lost-in-the-middle-reg-nb"
# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="checkpoint"
# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"
wandb.login(key="372f5c298afc4be9b40dd7b97523d394c3d30d05")
run = wandb.init(project="lost-in-the-middle-reg", name=f"loss_regularization_mu_{mu_str}")

# Funzione per calcolare la lunghezza degli input
def compute_input_lengths(examples):
    input_texts = formatting_prompts_func(examples)  # Usa la funzione che formatta i prompt
    tokenized_inputs = tokenizer(input_texts, truncation=False)  # Tokenizza senza troncamento
    lengths = [len(input_ids) for input_ids in tokenized_inputs['input_ids']]
    return {"input_length": lengths}

# Applica la funzione al dataset per ottenere le lunghezze degli input
dataset_with_lengths = dataset.map(compute_input_lengths, batched=True)

# Trova la lunghezza massima e media
max_length = max(dataset_with_lengths["input_length"])
average_length = sum(dataset_with_lengths["input_length"]) / len(dataset_with_lengths["input_length"])

print(f"La lunghezza massima di un input nel dataset è: {max_length} token")
print(f"La lunghezza media degli input nel dataset è: {average_length:.2f} token")

os.environ["CUDA_LAUNCH_BLOCKING"]="1"

trainer = LITMTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = eval_dataset,
    max_seq_length = max_seq_length,
    # dataset_num_proc = 2,
    packing = False, # Packs short sequences together to save time!
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    peft_config=peft_config,
    mu=mu,
    enable_litm=enable_litm,
    run_wandb=run,
    args = TrainingArguments(
        report_to="wandb",
        run_name= f"loss_regularization_mu_{mu_str}",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        # Se steps < 4000 -> 0.1, se 20.000 ->
        warmup_ratio = 0.1,
        num_train_epochs = 1,
        learning_rate = 2e-5,
        # learning_rate = 1e-4,
        # fp16 = not is_bfloat16_supported(),
        # bf16 = is_bfloat16_supported(),
        fp16 = False,
        bf16 = True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs = {"use_reentrant": True},
        optim = "adamw_8bit",
        # optim = "adamw_4bit",
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
# alpaca_prompt = Copied from above
# index = 1
# FastLanguageModel.for_inference(model) # Enable native 2x faster inference
# inputs = tokenizer(
# [
#     eval_dataset[index]["prompt"]
# ], return_tensors = "pt").to("cuda")

# outputs = model.generate(**inputs, use_cache = True)
# print(tokenizer.batch_decode(outputs)[0])
# print("-------------")
# print("Risposta esatta: ", eval_dataset[index]["completion"])

# ## SAVE
# model.save_pretrained(checkpoint_dir + "/litm_model_reg") # Local saving
# tokenizer.save_pretrained(checkpoint_dir + "/litm_model_reg")
# model.push_to_hub("Paoloc99/litm_model_reg", token="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") # Online saving
# tokenizer.push_to_hub("Paoloc99/litm_model_reg", token="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") # Online saving