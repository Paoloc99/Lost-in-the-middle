### Updating Unsloth without dependency updates
Use `pip install --upgrade --force-reinstall --no-cache-dir git+https://github.com/unslothai/unsloth.git`

### Loading LoRA adapters for continued finetuning
If you saved a LoRA adapter through Unsloth, you can also continue training using your LoRA weights. The optimizer state will be reset as well. To load even optimizer states to continue finetuning, see the next section.
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "LORA_MODEL_NAME",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
trainer = Trainer(...)
trainer.train()
```

### Finetuning from your last checkpoint
You must edit the `Trainer` first to add `save_strategy` and `save_steps`. Below saves a checkpoint every 50 steps to the folder `outputs`.
```python
trainer = SFTTrainer(
    ....
    args = TrainingArguments(
        ....
        output_dir = "outputs",
        save_strategy = "steps",
        save_steps = 50,
    ),
)
```
Then in the trainer do:
```python
trainer_stats = trainer.train(resume_from_checkpoint = True)
```
Which will start from the latest checkpoint and continue training.

### Saving models to 16bit for VLLM
To save to 16bit for VLLM, use:
```python
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")
```
To merge to 4bit to load on HuggingFace, first call `merged_4bit`. Then use `merged_4bit_forced` if you are certain you want to merge to 4bit. I highly discourage you, unless you know what you are going to do with the 4bit model (ie for DPO training for eg or for HuggingFace's online inference engine)
```python
model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")
```
To save just the LoRA adapters, either use:
```python
model.save_pretrained(...) AND tokenizer.save_pretrained(...)
```
Or just use our builtin function to do that:
```python
model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")
```

### Saving to GGUF

To save to GGUF, use the below to save locally:
```python
model.save_pretrained_gguf("dir", tokenizer, quantization_method = "q4_k_m")
model.save_pretrained_gguf("dir", tokenizer, quantization_method = "q8_0")
model.save_pretrained_gguf("dir", tokenizer, quantization_method = "f16")
```
For to push to hub:
```python
model.push_to_hub_gguf("hf_username/dir", tokenizer, quantization_method = "q4_k_m")
model.push_to_hub_gguf("hf_username/dir", tokenizer, quantization_method = "q8_0")
```
All supported quantization options for `quantization_method` are listed below:
```python
# https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp#L19
# From https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html
ALLOWED_QUANTS = \
{
    "not_quantized"  : "Recommended. Fast conversion. Slow inference, big files.",
    "fast_quantized" : "Recommended. Fast conversion. OK inference, OK file size.",
    "quantized"      : "Recommended. Slow conversion. Fast inference, small files.",
    "f32"     : "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
    "f16"     : "Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "q8_0"    : "Fast conversion. High resource use, but generally acceptable.",
    "q4_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q5_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q2_k"    : "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    "q3_k_l"  : "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_m"  : "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_s"  : "Uses Q3_K for all tensors",
    "q4_0"    : "Original quant method, 4-bit.",
    "q4_1"    : "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
    "q4_k_s"  : "Uses Q4_K for all tensors",
    "q4_k"    : "alias for q4_k_m",
    "q5_k"    : "alias for q5_k_m",
    "q5_0"    : "Higher accuracy, higher resource usage and slower inference.",
    "q5_1"    : "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s"  : "Uses Q5_K for all tensors",
    "q6_k"    : "Uses Q8_K for all tensors",
    "iq2_xxs" : "2.06 bpw quantization",
    "iq2_xs"  : "2.31 bpw quantization",
    "iq3_xxs" : "3.06 bpw quantization",
    "q3_k_xs" : "3-bit extra small quantization",
}
```

### Evaluation Loop - also OOM or crashing.
Set the trainer settings for evaluation to:
```python
SFTTrainer(
    args = TrainingArguments(
        fp16_full_eval = True,
        per_device_eval_batch_size = 2,
        eval_accumulation_steps = 4,
        evaluation_strategy = "steps",
        eval_steps = 1,
    ),
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
```
This will cause no OOMs and make it somewhat faster with no upcasting to float32.

### Chat Templates
Assuming your dataset is a list of list of dictionaries like the below:
```python
[
    [{'from': 'human', 'value': 'Hi there!'},
     {'from': 'gpt', 'value': 'Hi how can I help?'},
     {'from': 'human', 'value': 'What is 2+2?'}],
    [{'from': 'human', 'value': 'What's your name?'},
     {'from': 'gpt', 'value': 'I'm Daniel!'},
     {'from': 'human', 'value': 'Ok! Nice!'},
     {'from': 'gpt', 'value': 'What can I do for you?'},
     {'from': 'human', 'value': 'Oh nothing :)'},],
]
```
You can use our `get_chat_template` to format it. Select `chat_template` to be any of `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth`, and use `mapping` to map the dictionary values `from`, `value` etc. `map_eos_token` allows you to map `<|im_end|>` to EOS without any training.
```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    map_eos_token = True, # Maps <|im_end|> to </s> instead
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("philschmid/guanaco-sharegpt-style", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

You can also make your own custom chat templates! For example our internal chat template we use is below. You must pass in a `tuple` of `(custom_template, eos_token)` where the `eos_token` must be used inside the template.
```python
unsloth_template = \
    "{{ bos_token }}"\
    "{{ 'You are a helpful assistant to the user\n' }}"\
    "{% endif %}"\
    "{% for message in messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ '>>> User: ' + message['content'] + '\n' }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ '>>> Assistant: ' + message['content'] + eos_token + '\n' }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ '>>> Assistant: ' }}"\
    "{% endif %}"
unsloth_eos_token = "eos_token"

tokenizer = get_chat_template(
    tokenizer,
    chat_template = (unsloth_template, unsloth_eos_token,), # You must provide a template and EOS token
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    map_eos_token = True, # Maps <|im_end|> to </s> instead
)
```

### 2x Faster Inference
Unsloth supports natively 2x faster inference. All QLoRA, LoRA and non LoRA inference paths are 2x faster. This requires no change of code or any new dependencies.
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 64)
```

### NotImplementedError: A UTF-8 locale is required. Got ANSI
See https://github.com/googlecolab/colabtools/issues/3409

In a new cell, run the below:
```python
import locale
locale.getpreferredencoding = lambda: "UTF-8"
```

# Native Windows Install (No WSL)

Installing Unsloth on native windows is possible, it just takes a few extra steps.

## Requirements

- Cuda 12.1 (Untested on 11.8 although it probably would work)
- Python 3.11
- Visual Studio 2022 build tools [^1^][1]

### Installing Visual Studio 2022 Build Tools

1. First, you must install the installer.
2. Open Visual Studio installer and select 'Desktop Development with C++' under the "Workload" tab.
3. Under the Individual Components Tab, search for these optional components if not automatically selected: "MSVC v143-VS 2022 C++ x64/x86 build tools, Windows 11 SDK, C++ CMake tools for windows, Testing tools core features-Build Tools, C++ AddressSanitizer".
4. Hit install and close Visual Studio 2022.

## Installing Premade Wheels and Triton Library

[Wheels and Library(https://drive.google.com/drive/folders/1aWSFb-ZR8TTIDdRlDBBCh-YvvCxmt6Bc)]

Note: I have made pre-made wheel Deepspeed (These wheels are experimental wheels, just because you have them installed does not mean all the calls made work on non-unix based OS, lucky for us Single GPU trainers, it does work on windows even when going into shared memory. I did not make the Triton windows wheels or windows library for Triton, they were made by wkpark on GitHub).

### Installing Triton

1. The first package you will need to install is Triton, as Deepspeed depends on it.
2. In order to install Triton, you will need the Triton Libraries. Unzip the Triton libraries to your C drive like so 'C:\llvm-5e5a22ca-windows-x64'.
3. Add The Bin and lib to your environment variables (same way you did when installing CUDA) so 'C:\llvm-5e5a22ca-windows-x64\bin' and 'C:\llvm-5e5a22ca-windows-x64\lib' to your system path inside the environment variables page.
4. Now you may install the wheels. Download the wheels and run 'pip install triton-2.1.0-cp311-cp311-win_amd64.whl'.

### Installing Deepspeed

1. Download the package and run 'pip install deepspeed-0.13.1+unknown-py3-none-any.whl'.

### Installing Latest BitsAndBytes Release Windows Package

1. Download the package and run 'pip install bitsandbytes-0.43.0.dev0-cp311-cp311-win_amd64.whl'.

### Install Xformers

1. Run 'pip install xformers'. Note: you might get issues running training related to Xformers. If you do, run 'pip uninstall xformers' then 'pip install xformers'.

## Install Unsloth

Install Unsloth as normal through pip. Instructions are on the GitHub home page. If during training you get an error, you might have to delete this line at line 1291 in C:\Users\Training\AppData\Local\Programs\Python\Python311\Lib\site-packages\unsloth\models\llama.py:

```python
for module in target_modules:
    assert(module in accepted_modules)
    pass  
```
Windows install instructions made by [Nicolas Mejia Petit (https://twitter.com/mejia_petit)]
