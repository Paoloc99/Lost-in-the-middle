import argparse
import torch
from transformers import TrainingArguments, EarlyStoppingCallback, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import re
import os
import datetime
from LITMTrainer import LITMTrainer
from peft import LoraConfig
from DataCollatorForLostInTheMiddle import DataCollatorForLostInTheMiddle
import wandb

WANDB_KEY = "372f5c298afc4be9b40dd7b97523d394c3d30d05"
HF_TOKEN = "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"

def main(args):
    # Configuration from arguments
    max_seq_length = args.max_seq_length
    dtype = torch.float16 if args.dtype == 'float16' else torch.bfloat16 if args.dtype == 'bfloat16' else None
    load_in_4bit = args.load_in_4bit
    enable_litm = args.enable_litm
    use_liger = args.use_liger
    gradient_checkpointing = args.gradient_checkpointing
    use_reentrant = args.use_reentrant

    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_train_epochs = args.num_train_epochs
    mu = args.mu

    mu_str = str(mu).replace(".", "_")
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
            use_cache= not gradient_checkpointing,
            use_reentrant=use_reentrant
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype,
            quantization_config=bnb_config,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    peft_config = LoraConfig(
        lora_alpha=4,
        lora_dropout=0,
        r=2,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        use_rslora=False,
        loftq_config=None,
    )

    # Load dataset
    train_dataset = load_dataset("Paoloc99/dataset", split=f"train[:{args.train_dataset_size}]")
    eval_dataset = load_dataset("Paoloc99/dataset", split="train[-300:]")

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        full_text = []
        if not isinstance(examples['prompt'], list):
            full_text = [f"{examples['prompt']} {examples['completion'].strip()}{EOS_TOKEN}"]
        else:
            for i in range(len(examples['prompt'])):
                full_text.append(f"{examples['prompt'][i]} {examples['completion'][i].strip()}{EOS_TOKEN}")
        return full_text
    
    # Definire la lunghezza massima consentita
    max_allowed_length = 1200  # Imposta qui la lunghezza massima desiderata

    # Funzione per calcolare la lunghezza di ciascun esempio
    def filter_long_inputs(example):
        input_text = formatting_prompts_func(example)  # Usa la funzione che formatta i prompt
        tokenized_input = tokenizer(input_text, truncation=False)  # Tokenizza senza troncamento
        input_length = len(tokenized_input['input_ids'][0])  # Lunghezza dell'input
        return input_length <= max_allowed_length  # Mantieni solo gli input con lunghezza <= max_allowed_length

    # Applicare il filtro al dataset
    train_dataset = train_dataset.filter(filter_long_inputs, batched=False)
    eval_dataset = eval_dataset.filter(filter_long_inputs, batched=False)
    print("Lunghezza train dataset dopo filter:", len(train_dataset))
    print("Lunghezza eval dataset dopo filter:", len(eval_dataset))

    response_template_with_context = "\nAnswer:"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[-2:]
    document_template_with_context = "\nDocument ["
    document_template_ids = tokenizer.encode(document_template_with_context, add_special_tokens=False)[-2:]
    question_template_with_context = "\nQuestion:"
    question_template_ids = tokenizer.encode(question_template_with_context, add_special_tokens=False)[-2:]

    collator = DataCollatorForLostInTheMiddle(
        response_template_ids,
        document_template=document_template_ids,
        question_template=question_template_ids,
        tokenizer=tokenizer
    )

    # Checkpoint handling
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d")
    checkpoint_dir = 'checkpoints'
    output_dir = f'checkpoints/{timestamp}'

    def get_latest_checkpoint_dir(base_dir):
        timestamp_dirs = [d for d in os.listdir(base_dir) if re.match(r'\d{4}-\d{2}-\d{2}', d)]
        timestamp_dirs.sort(key=lambda date: datetime.datetime.strptime(date, "%Y-%m-%d"), reverse=True)
        latest_timestamp_dir = timestamp_dirs[0] if timestamp_dirs else None
        return os.path.join(base_dir, latest_timestamp_dir) if latest_timestamp_dir else None

    latest_timestamp_dir = get_latest_checkpoint_dir(checkpoint_dir)

    def get_latest_checkpoint(checkpoint_dir):
        if checkpoint_dir is None:
            return None
        checkpoint_dirs = [d for d in os.listdir(checkpoint_dir) if re.match(r'checkpoint-\d+', d)]
        checkpoint_dirs.sort(key=lambda x: int(re.search(r'\d+', x).group()), reverse=True)
        latest_checkpoint_dir = checkpoint_dirs[0] if checkpoint_dirs else None
        return os.path.join(checkpoint_dir, latest_checkpoint_dir) if latest_checkpoint_dir else None

    checkpoint_path = get_latest_checkpoint(latest_timestamp_dir)
    print(f"Checkpoint {checkpoint_path} found." if checkpoint_path and os.path.exists(checkpoint_path) else "No checkpoint found.")

    # Trainer setup
    os.environ["WANDB_PROJECT"] = "lost-in-the-middle-reg"
    os.environ["WANDB_NOTEBOOK_NAME"] = "lost-in-the-middle-reg-nb"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_WATCH"] = "false"
    wandb.login(key=WANDB_KEY)
    run = wandb.init(project="lost-in-the-middle-reg", name=f"loss_regularization_mu_{mu_str}")

    # Funzione per calcolare la lunghezza degli input
    # def compute_input_lengths(examples):
    #     input_texts = formatting_prompts_func(examples)  # Usa la funzione che formatta i prompt
    #     tokenized_inputs = tokenizer(input_texts, truncation=False)  # Tokenizza senza troncamento
    #     lengths = [len(input_ids) for input_ids in tokenized_inputs['input_ids']]
    #     return {"input_length": lengths}

    # # Applica la funzione al dataset per ottenere le lunghezze degli input
    # dataset_with_lengths = train_dataset.map(compute_input_lengths, batched=True)

    # Trova la lunghezza massima e media
    # 2975 tokens e 1296.78 token
    # max_length = max(dataset_with_lengths["input_length"])
    # average_length = sum(dataset_with_lengths["input_length"]) / len(dataset_with_lengths["input_length"])

    # print(f"La lunghezza massima di un input nel dataset è: {max_length} token")
    # print(f"La lunghezza media degli input nel dataset è: {average_length:.2f} token")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    trainer = LITMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=max_seq_length,
        packing=False,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        peft_config=peft_config,
        mu=mu,
        enable_litm=enable_litm,
        run_wandb=run,
        args=TrainingArguments(
            report_to="wandb",
            run_name=f"loss_regularization_mu_{mu_str}",
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.1,
            num_train_epochs=num_train_epochs,
            learning_rate=2e-5,
            fp16=False,
            bf16=True,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
            optim="adamw_8bit",
            weight_decay=0.1,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            logging_steps=5,
            log_level='debug',
            save_steps=100,
            save_total_limit=10,
            save_strategy="steps",
            eval_steps=100,
            eval_strategy="steps",
            do_eval=False,
            load_best_model_at_end=True
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train(resume_from_checkpoint=checkpoint_path)

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Inference
    index = 1
    inputs = tokenizer([eval_dataset[index]["prompt"]], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, use_cache=True)
    print(tokenizer.batch_decode(outputs)[0])
    print("-------------")
    print("Risposta esatta: ", eval_dataset[index]["completion"])

    # Save model
    model.save_pretrained(checkpoint_dir + "/litm_model_reg_"+ mu_str)
    tokenizer.save_pretrained(checkpoint_dir + "/litm_model_reg_"+ mu_str)
    model.push_to_hub(f"Paoloc99/litm_model_reg_{mu_str}", token=HF_TOKEN)
    tokenizer.push_to_hub(f"Paoloc99/litm_model_reg_{mu_str}", token=HF_TOKEN)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training script with specified parameters.")
    parser.add_argument('--max_seq_length', type=int, default=4096, help='Maximum sequence length.')
    parser.add_argument('--dtype', type=str, default='None', choices=['float16', 'bfloat16', 'None'], help='Data type to use.')
    parser.add_argument('--load_in_4bit', type=bool, default=True, help='Whether to load model in 4-bit.')
    parser.add_argument('--enable_litm', type=bool, default=True, help='Whether to enable Lost in the Middle.')
    parser.add_argument('--use_liger', type=bool, default=False, help='Whether to use LIGER kernel.')
    parser.add_argument('--gradient_checkpointing', type=bool, default=False, help='Whether to use Gradient Checkpointing.')
    parser.add_argument('--use_reentrant', type=bool, default=False, help='Whether to use Reentrant.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4, help='Batch size per device during training.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Number of gradient accumulation steps.')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--mu', type=float, default=10000, help='Regularization coefficient.')
    parser.add_argument('--train_dataset_size', type=int, default=70000, help='Size of the training dataset.')
    parser.add_argument('--model_name', type=str, default='unsloth/tinyllama-bnb-4bit', help='Model name.')

    args = parser.parse_args()
    main(args)
