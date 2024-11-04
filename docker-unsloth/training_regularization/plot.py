import matplotlib.font_manager as font_manager
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
import gc
import polars as pl

def download_dataset(dataset_type, position):
    # Crea la cartella 'data' se non esiste
    os.makedirs('data', exist_ok=True)

    # Mapping dei link per i dataset
    dataset_links = {
        "random": {
            0: "1ZfiEmnqcBa1lZ2YPz-Cj7RecmNZCMjNN",
            3: "1uD9jy5EVBoLg1v8mMzr5sAswM8ykIBoN",
            7: "1y7xG2L7s0B5DrX11kuNHZ4Jf_W2zn_gA",
            "random": "1ky80V0A6TtaFwzOz3Om2Il6X8pWRta27"
        },
        "related": {
            0: "17424KW_SJ6a5kNuEPkleuaBOzwxoW7hh",
            3: "1fcY4rR9R588o4IySK7x6Y0l6br7Gi-vj",
            7: "1ZiheY6eWtcAJUZxhxRBTvoF6JmwGuvPd",
            "random": "1FXsnYEhUgdzV8eDwhisG0LacTORULFqK"
        }
    }

    # Controlla se i parametri sono validi
    if dataset_type not in dataset_links:
        raise ValueError(f"Tipo di dataset non valido: {dataset_type}. Usa 'random' o 'related'.")

    if position not in dataset_links[dataset_type]:
        raise ValueError(f"Posizione non valida: {position}. Usa 0, 3, 7 o 'random'.")

    # Ottieni il link corretto per il download
    link_id = dataset_links[dataset_type][position]
    # print(link_id)
    # download_command = f"cd data"

    # # Esegui il comando di download
    # os.system(download_command)

    # Costruisci il nome del file in base ai parametri
    file_name = f"{dataset_type}_dataset_gold_at_{position}.parquet" if position != "random" else f"{dataset_type}_dataset_gold_at_random_position.parquet"
    
    # Carica il dataset
    df = pl.read_parquet(f'data/{file_name}')

    print(file_name)
    # Stampa per visualizzare il dataset (o gestiscilo come necessario)
    return df

@torch.no_grad()
def get_documents_attention_to_answer(generated_string, tokenizer, model):
    output_tokenized = tokenizer(
        generated_string,
        padding=True,
        truncation=True,
        max_length=4096,
        return_tensors="pt"
    ).to('cuda:0')
    outputs = model(
        **output_tokenized,
        output_attentions=True,
    )
    def find_doc_positions(generated_string):
        pattern = r"Document \[\d+?\].*?$"
        matches = re.finditer(pattern, generated_string, re.MULTILINE)

        documents_positions = [(match.start(), match.end()) for match in matches]
        return [(output_tokenized.char_to_token(i), output_tokenized.char_to_token(j)) for i, j in documents_positions]

    def find_answer_potitions(generated_string):
        # answer position ranges
        pattern = r"Answer:.*?$"

        matches = re.finditer(pattern, generated_string, re.MULTILINE)
        answer_positions = [(match.start() + len("Answer:"), min(match.end(), len(generated_string)-1)) for match in matches]
        assert len(answer_positions) == 1, "More than one answer found"
        answer_positions = answer_positions[0]
        return output_tokenized.char_to_token(answer_positions[0]), output_tokenized.char_to_token(answer_positions[1])

    document_token_positions = find_doc_positions(generated_string)
    try:
      answer_token_positions = find_answer_potitions(generated_string)

      document_attentions_to_answer = []
      for attention_layer in range(len(outputs.attentions)):
          attention_np = outputs.attentions[attention_layer].float().mean(1).squeeze(0).cpu().numpy()

          answer_attention = attention_np[answer_token_positions[0]: answer_token_positions[1]].mean(0)
          doc_avgs = np.array([answer_attention[i:j].mean() for i, j in document_token_positions])
          normalized_doc_avgs = doc_avgs / doc_avgs.sum()
          document_attentions_to_answer.append(normalized_doc_avgs)

      document_attentions_to_answer = np.stack(document_attentions_to_answer)

      # Let's avoid the VRAM to explode
      del outputs
      gc.collect()
      torch.cuda.empty_cache()
      return document_attentions_to_answer
    except Exception as e:
      del outputs
      gc.collect()
      torch.cuda.empty_cache()
      raise e
    
def plot_attentions(document_attentions_to_answer, title=None, save=True, filename=None):
    num_layers, num_docs = document_attentions_to_answer.shape

    # Plotting
    plt.figure(figsize=(10,8))

    ticks = [f'Doc_{i}' for i in range(num_docs)]

    y_ticks_labels = [str(i) for i in range(1, num_layers + 2, 4)]
    y_ticks_labels[-1] = str(num_layers)
    y_ticks_positions = [i-0.5 for i in range(1, num_layers + 2, 4)]
    # y_ticks_positions[-1] = 31.5
    y_ticks_positions[-1] = float(num_layers) - 0.5

    sns.heatmap(document_attentions_to_answer, annot=False, cmap='Blues', xticklabels=ticks, vmin=0, vmax=val_max)
    # sns.heatmap(document_attentions_to_answer, annot=False, cmap='Blues', xticklabels=ticks)

    plt.xticks(rotation=20)
    plt.yticks(rotation=0)
    plt.xlabel("Documents in Context")
    plt.yticks(y_ticks_positions, y_ticks_labels)
    plt.ylabel("Attention Layers")
    title = f"{title}" if title else ""
    plt.title(title)

    plt.tight_layout()
    if save:
        os.makedirs("figures", exist_ok=True)
        if filename:
            plt.savefig(f"figures/{filename}", dpi=600)
        else:
            plt.savefig(f"figures/{dataset_type}_dataset_gold_at_{position}.png", dpi=600)
    else:
        plt.show()

# model_name = "microsoft/Phi-3-mini-128k-instruct"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "Qwen/Qwen2-0.5B-Instruct"
# model_name = "Qwen/Qwen2-1.5B-Instruct"
# model_name = "unsloth/tinyllama"
# model_name =  "Paoloc99/litm_model"
model_name =  "Paoloc99/litm_model_1300"
# model_name = "Paoloc99/litm_model_reg_10000_0"
# model_name = "Paoloc99/litm_model_reg_1000_0"
# model_name = "Paoloc99/litm_model_new_reg_1_0"
# model_name = "Paoloc99/litm_model_new_reg_100_0"


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# dataset_type = 'random'
dataset_type = 'related'
position = 'random' 

df = download_dataset(dataset_type, position)

from tqdm.auto import tqdm
import random


def calculate_attentions(df):
    documents_attention_to_answer = []
    gold_position_counts = torch.zeros(8)
    for index in tqdm(range(500)):
        row = df[index]

        documents = row['Documents'][0]
        # Mescola i documenti
        documents.shuffle()

        prompt = "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES.\nDocuments:\n"
        for i, document in enumerate(documents):
            prompt = f"{prompt}Document [{i}](Title: {document['Title']}) {document['Text']}\n"

        prompt = f"{prompt}Question: {row['Question'][0]}\n"
        prompt = f"{prompt}Answer:"

        inputs = tokenizer(prompt, return_tensors="pt")
        device = torch.device('cuda')
        inputs = inputs.to(device)
        # print(inputs['input_ids'].shape, inputs['attention_mask'].shape)
        output = model.generate(**inputs, max_new_tokens=10)

        output = output.cpu().squeeze() # this contains both prompt (with documents) and answers
        generated_string = tokenizer.decode(output, skip_special_tokens=True)
        # print(generated_string)

        documents_attention_to_answer.append(get_documents_attention_to_answer(generated_string, tokenizer, model))

        gold_position = row['Golden_idx']
        gold_position_counts[gold_position] += 1
    return documents_attention_to_answer, gold_position_counts

def expected_matrix(x: torch.Tensor,
                    rows: int) -> torch.Tensor:
  """
  """
  # Normalize the original tensor
  x = x/x.sum()

  return x.unsqueeze(0).repeat(rows, 1)

documents_attention_to_answer, gold_position_counts = calculate_attentions(df)

# cleaned = [doc for doc in documents_attention_to_answer if doc.shape == (32, 8) and not np.isnan(doc).any()]
cleaned = [doc for doc in documents_attention_to_answer if doc.shape == (24, 8) and not np.isnan(doc).any()]
# cleaned = [doc for doc in documents_attention_to_answer if doc.shape == (22, 8) and not np.isnan(doc).any()]

np_stack = np.stack(cleaned)
# np_stack = np.stack(documents_attention_to_answer)
mean_attentions = np_stack.mean(0)
mean_attentions.shape

val_max = np.max(mean_attentions)

print("Gold position counts:", gold_position_counts)

normalized_matrix = expected_matrix(gold_position_counts, rows=mean_attentions.shape[0])

mu = "10000"
filename = f"litm_1300"

plot_attentions(mean_attentions, title=f"Attention from Answer to Documents", save=True, filename=filename)
print({"dataset_type": dataset_type, "position": position})

filename1 = f"{filename}_expected"
plot_attentions(normalized_matrix, title=f"Attention from Answer to Documents", save=True, filename=filename1)

filename2 = f"{filename}_diff"
plot_attentions((normalized_matrix-mean_attentions)**2, title=f"Attention from Answer to Documents", save=True, filename=filename2)

print(torch.mean((normalized_matrix-mean_attentions)**2))

dataset_type = 'random'
position = 'random' 

df = download_dataset(dataset_type, position)

documents_attention_to_answer, gold_position_counts = calculate_attentions(df)

cleaned = [doc for doc in documents_attention_to_answer if doc.shape == (24, 8) and not np.isnan(doc).any()]

np_stack = np.stack(cleaned)
mean_attentions = np_stack.mean(0)
mean_attentions.shape

val_max = np.max(mean_attentions)

print("Gold position counts:", gold_position_counts)

normalized_matrix = expected_matrix(gold_position_counts, rows=mean_attentions.shape[0])

filename_random = f"{filename}_random"

plot_attentions(mean_attentions, title=f"Attention from Answer to Documents", save=True, filename=filename_random)
print({"dataset_type": dataset_type, "position": position})

filename1 = f"{filename_random}_expected"
plot_attentions(normalized_matrix, title=f"Attention from Answer to Documents", save=True, filename=filename1)

filename2 = f"{filename_random}_diff"
plot_attentions((normalized_matrix-mean_attentions)**2, title=f"Attention from Answer to Documents", save=True, filename=filename2)

print(torch.mean((normalized_matrix-mean_attentions)**2))