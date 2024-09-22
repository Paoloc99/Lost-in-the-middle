from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import load_dataset
import datetime
import re
import os

max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

## MODEL
model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/tinyllama", 
    model_name = "unsloth/tinyllama-bnb-4bit", # for 16bit loading
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 2,#32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 4,#32,
    lora_dropout = 0, # Currently only supports dropout = 0
    bias = "none",    # Currently only supports bias = "none"
    # use_gradient_checkpointing = False, # @@@ IF YOU GET OUT OF MEMORY - set to True @@@
    use_gradient_checkpointing = True, # @@@ IF YOU GET OUT OF MEMORY - set to True @@@
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

## DATASET
def formatting_prompts_func(examples):
  full_text = []
  if not isinstance(examples['prompt'], list):
    # print("Non lista:", len(examples['prompt']))
    full_text = [f"{examples['prompt']} {examples['completion']}"]
  else:
    # print(len(examples['prompt']))
    for i in range(len(examples['prompt'])):
      full_text.append(f"{examples['prompt'][i]} {examples['completion'][i].strip()}")
  return full_text#{ "text" : full_text, }

dataset = load_dataset("Paoloc99/dataset", split="train[:15000]")
eval_dataset = load_dataset("Paoloc99/dataset", split="train[-1000:]")

response_template_with_context = "\nAnswer:"  # We added context here: "\n". This is enough for this tokenizer
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[22550, 29901]`
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

## CHECKPOINT PATH
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d")

checkpoint_dir = 'checkpoints/'
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
print(checkpoint_path)

## TRAINER
if checkpoint_path and not os.path.exists(checkpoint_path):
    print(f"Checkpoint {checkpoint_path} does not exist.")
else:
    print(f"Checkpoint {checkpoint_path} found.")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    # eval_dataset = val_dataset,
    #dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Packs short sequences together to save time!
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.1,
        num_train_epochs = 1,
        learning_rate = 2e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim = "adamw_8bit",
        weight_decay = 0.1,
        lr_scheduler_type = "cosine",#"linear",
        seed = 3407,
        output_dir = output_dir,
        # eval_steps=5,
        # eval_strategy="steps",
        logging_steps=5,
        log_level='debug', #'info',
        save_steps=100,
        save_total_limit=5,
        # resume_from_checkpoint=checkpoint_path
    ),
)

## STATS PRE
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

## TRAIN
trainer_stats = trainer.train(resume_from_checkpoint=checkpoint_path)

## STATS POST
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

## TEST INFERENCE
print(eval_dataset[2]["prompt"])
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    eval_dataset[2]["prompt"]
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
print(tokenizer.batch_decode(outputs))

# from transformers import TextStreamer
# text_streamer = TextStreamer(tokenizer)
# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

print("Risposta esatta: ", eval_dataset[2]["completion"])