import random
from trl import SFTTrainer
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from peft import PeftConfig
from functools import wraps
# from unsloth import FastLanguageModel
import gc
from torch.nn.functional import kl_div
import wandb
from wandb.sdk.wandb_run import Run

class LITMTrainer(SFTTrainer):
    r"""
    Class definition of the Supervised Finetuning Trainer (SFT Trainer).
    This class is a wrapper around the `transformers.Trainer` class and inherits all of its attributes and methods.
    The trainer takes care of properly initializing the PeftModel in case a user passes a `PeftConfig` object.

    Args:
        model (Union[`transformers.PreTrainedModel`, `nn.Module`, `str`]):
            The model to train, can be a `PreTrainedModel`, a `torch.nn.Module` or a string with the model name to
            load from cache or download. The model can be also converted to a `PeftModel` if a `PeftConfig` object is
            passed to the `peft_config` argument.
        args (Optional[`transformers.TrainingArguments`]):
            The arguments to tweak for training. Please refer to the official documentation of `transformers.TrainingArguments`
            for more information.
        data_collator (Optional[`transformers.DataCollator`]):
            The data collator to use for training.
        train_dataset (Optional[`datasets.Dataset`]):
            The dataset to use for training. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        eval_dataset (Optional[Union[`datasets.Dataset`, Dict[`str`, `datasets.Dataset`]]]):
            The dataset to use for evaluation. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        tokenizer (Optional[`transformers.PreTrainedTokenizer`]):
            The tokenizer to use for training. If not specified, the tokenizer associated to the model will be used.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to None):
            The function used to compute metrics during evaluation. It should return a dictionary mapping metric names to metric values.
            If not specified, only the loss will be computed during evaluation.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Optional[PeftConfig]`):
            The PeftConfig object to use to initialize the PeftModel.
        dataset_text_field (`Optional[str]`):
            The name of the text field of the dataset, in case this is passed by a user, the trainer will automatically create a
            `ConstantLengthDataset` based on the `dataset_text_field` argument.
        formatting_func (`Optional[Callable]`):
            The formatting function to be used for creating the `ConstantLengthDataset`.
        max_seq_length (`Optional[int]`):
            The maximum sequence length to use for the `ConstantLengthDataset` and for automatically creating the Dataset. Defaults to `512`.
        infinite (`Optional[bool]`):
            Whether to use an infinite dataset or not. Defaults to `False`.
        num_of_sequences (`Optional[int]`):
            The number of sequences to use for the `ConstantLengthDataset`. Defaults to `1024`.
        chars_per_token (`Optional[float]`):
            The number of characters per token to use for the `ConstantLengthDataset`. Defaults to `3.6`. You can check how this is computed in the
            stack-llama example: https://github.com/huggingface/trl/blob/08f550674c553c36c51d1027613c29f14f3676a5/examples/stack_llama/scripts/supervised_finetuning.py#L53.
        packing (`Optional[bool]`):
            Used only in case `dataset_text_field` is passed. This argument is used by the `ConstantLengthDataset` to pack the sequences
            of the dataset.
        dataset_num_proc (`Optional[int]`):
            The number of workers to use to tokenize the data. Only used when `packing=False`. Defaults to None.
        dataset_batch_size (`int`):
            The number of examples to tokenize per batch. If batch_size <= 0 or batch_size == None,
            tokenize the full dataset as a single batch. Defaults to 1000.
        neftune_noise_alpha (`Optional[float]`):
            If not `None`, this will activate NEFTune noise embeddings. This has been proven to drastically improve model performances for instruction
            fine-tuning. Check out the original paper here: https://arxiv.org/abs/2310.05914 and the original code here: https://github.com/neelsjain/NEFTune
        model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        dataset_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when creating packed or non-packed datasets
        eval_packing: (`Optional[bool]`, *optional*):
            Whether to pack the eval dataset as well. Defaults to `packing` if `None` is passed.
        mu: (`Optional[float]`, *optional*):
            Value for the mu value in the KL divergence. Defaults to `1.0`.
        enable_litm: (`Optional[bool]`, *optional*):
            Enable or not the Lost in The Middle regularization. Default to True.
        run_wandb: (`Optional[]`, *optional*):
            Run for wandb in order to log computed loss. Default to None.
    """
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        dataset_text_field: Optional[str] = None,
        packing: Optional[bool] = False,
        formatting_func: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        infinite: Optional[bool] = None,
        num_of_sequences: Optional[int] = 1024,
        chars_per_token: Optional[float] = 3.6,
        dataset_num_proc: Optional[int] = None,
        dataset_batch_size: int = 1000,
        neftune_noise_alpha: Optional[float] = None,
        model_init_kwargs: Optional[Dict] = None,
        dataset_kwargs: Optional[Dict] = None,
        eval_packing: Optional[bool] = None,
        mu: Optional[float] = 1.0,
        enable_litm: Optional[bool] = True,
        run_wandb: Optional[Run] = None
    ):  
        self.tokenizer = tokenizer
        self.mu = mu
        self.enable_litm = enable_litm
        self.run_wandb = run_wandb

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config= peft_config,
            dataset_text_field=dataset_text_field,
            packing=packing,
            formatting_func=formatting_func,
            max_seq_length=max_seq_length,
            infinite=infinite,
            num_of_sequences=num_of_sequences,
            chars_per_token=chars_per_token,
            dataset_num_proc=dataset_num_proc,
            dataset_batch_size=dataset_batch_size,
            neftune_noise_alpha=neftune_noise_alpha,
            model_init_kwargs=model_init_kwargs,
            dataset_kwargs=dataset_kwargs,
            eval_packing=eval_packing
        )

    def shuffle_documents(self, inputs, document_start_positions, document_end_positions, answer_positions):
        # Lista per i nuovi inputs e mappa delle posizioni
        new_inputs = []
        new_attention_mask = []
        new_labels = []
        document_maps = []
        new_document_start_positions = []
        new_document_end_positions = []
        new_answer_positions = []

        # Itera attraverso ogni elemento del batch
        for batch_idx in range(len(inputs['input_ids'])):
            input_ids = inputs['input_ids'][batch_idx]
            start_positions = document_start_positions[batch_idx]
            end_positions = document_end_positions[batch_idx]
            
            # Costruisci una lista di tuple (start, end) per i documenti
            documents = list(zip(start_positions, end_positions))
            
            # Genera una lista di indici dei documenti e mescolali per creare la mappa di shuffle
            indices = list(range(len(documents)))
            shuffled_indices = indices[:]
            random.shuffle(shuffled_indices)
            
            # Crea la mappa vecchia->nuova posizione
            shuffle_map = {old_idx: new_idx for old_idx, new_idx in zip(indices, shuffled_indices)}
            original_index_map = {original_idx: new_idx for new_idx, original_idx in enumerate(indices)}
            
            # Crea un nuovo input_ids sostituendo i documenti secondo la mappa di shuffle
            new_input_ids = []  # Copia l'input corrente
            previous_end = 0
            current_pos = 0
            new_start_positions = []
            new_end_positions = []

            for old_idx, new_idx in shuffle_map.items():
                
                # Ottieni le posizioni del documento originale e del documento da sostituire
                old_start, old_end = documents[old_idx]
                new_start, new_end = documents[new_idx]
                
                if old_idx == 0:
                    new_input_ids.extend(input_ids[:old_start])
                else:
                    new_input_ids.extend(input_ids[previous_end:old_start])
                
                current_pos = len(new_input_ids)
                new_start_positions.append(current_pos)
                
                replacement_document = input_ids[new_start:new_end + 1]
                
                new_input_ids.extend(replacement_document)

                current_pos = len(new_input_ids)
                new_end_positions.append(current_pos)

                previous_end = old_end+1

                if old_idx == len(shuffle_map)-1:
                    new_input_ids.extend(input_ids[old_end+1:len(input_ids)])

            device = input_ids.device.type
            new_input_ids = torch.Tensor(new_input_ids).to(torch.int64).to(device)
            new_start_positions = torch.Tensor(new_start_positions).to(torch.int64).to(device)
            new_end_positions = torch.Tensor(new_end_positions).to(torch.int64).to(device)
            new_inputs.append(input_ids)
            new_inputs.append(new_input_ids)
            document_maps.append(original_index_map)
            document_maps.append(shuffle_map)
            new_document_start_positions.append(start_positions)
            new_document_start_positions.append(new_start_positions)
            new_document_end_positions.append(end_positions)
            new_document_end_positions.append(new_end_positions)
            new_attention_mask.append(inputs['attention_mask'][batch_idx])
            new_attention_mask.append(inputs['attention_mask'][batch_idx])
            new_labels.append(inputs['labels'][batch_idx])
            new_labels.append(inputs['labels'][batch_idx])
            new_answer_positions.append(answer_positions[batch_idx].to(torch.int64))
            new_answer_positions.append(answer_positions[batch_idx].to(torch.int64))

        inputs['input_ids'] = torch.stack(new_inputs)
        inputs['attention_mask'] = torch.stack(new_attention_mask)
        inputs['labels'] = torch.stack(new_labels)
        return inputs, document_maps, new_document_start_positions, new_document_end_positions, new_answer_positions

    def get_documents_attention_to_answer(self, attentions, document_start_positions, document_end_positions, answer_position, batch_idx):
        try:
            document_attentions_to_answer = []
            device = attentions[0].device.type
            # device = "cpu"
            for attention_layer in range(len(attentions)):
                attention_np = attentions[attention_layer][batch_idx].float().mean(0)
                answer_attention = attention_np[answer_position:len(attention_np)-1].mean(0)
                doc_avgs = torch.stack([answer_attention[i:j].mean() for i, j in zip(document_start_positions, document_end_positions)])
                normalized_doc_avgs = doc_avgs / doc_avgs.sum()
                document_attentions_to_answer.append(normalized_doc_avgs)

            document_attentions_to_answer = torch.stack(document_attentions_to_answer).to(device)

            # Let's avoid the VRAM to explode
            gc.collect()
            torch.cuda.empty_cache()
            return document_attentions_to_answer
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            raise e

    def kl_regularizer(self, x, batch_documents_map):
        """Regularizer based on the Kullback-Leibler divergence Loss.
        """
        def normalize(x):
            return torch.log(x/x.sum())
        
        def normalize_softmax(x):
            return torch.log_softmax(x,dim=1)
        
        kl_loss = 0
        batch_size, layers, docs = x.shape

        for b in range(0, batch_size, 2):
            original = x[b, :, :].unsqueeze(0)
            shuffled = x[b+1, :, :].unsqueeze(0)
            original_map = batch_documents_map[b]
            shuffled_map = batch_documents_map[b+1]

            for key in original_map:
                # Reduction over a fake batch of 1
                kl_loss += kl_div(normalize_softmax(original[ :, :, key]), normalize_softmax(shuffled[ :, :, shuffled_map[key]]), reduction='batchmean', log_target=True)

        print("KL_LOSS senza normalizzazione per docs: ", kl_loss / layers)
        kl_loss /= docs*layers
        print("KL_LOSS con normalizzazione per docs: ", kl_loss)
        return kl_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if "answer_position" in inputs:
            answer_position = inputs.pop("answer_position")
        else:
            answer_position = None

        if "document_start_positions" in inputs:
            document_start_positions = inputs.pop("document_start_positions")
        else:
            document_start_positions = None

        if "document_end_positions" in inputs:
            document_end_positions = inputs.pop("document_end_positions")
        else:
            document_end_positions = None

        if self.enable_litm:
            # Pulire input da indicazioni su posizioni
            inputs, document_maps, new_document_start_positions, new_document_end_positions, new_answer_positions = self.shuffle_documents(inputs, document_start_positions, document_end_positions, answer_position) 
        
        print("Lunghezza batches: ", inputs['input_ids'].shape)
        outputs = model(**inputs, output_attentions=True)

        if self.enable_litm:
            attentions = outputs["attentions"]

            documents_attention_to_answer = []

            for batch_index in range(outputs["logits"].shape[0]):
                documents_attention_to_answer.append(self.get_documents_attention_to_answer(attentions, new_document_start_positions[batch_index], new_document_end_positions[batch_index], new_answer_positions[batch_index], batch_index))
            cleaned = [doc for doc in documents_attention_to_answer if doc.shape == (len(attentions), len(new_document_start_positions[0])) and not torch.isnan(doc).any()]
            attentions_tensor = torch.stack(cleaned)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if self._is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in self.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            model_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if self.enable_litm:
                kl_loss = self.kl_regularizer(attentions_tensor, document_maps)
                if self.run_wandb:
                    print("model_loss: ",model_loss.item())
                    print("kl_loss: ", kl_loss.item())
                    self.run_wandb.log({"model_loss":model_loss.item(), "kl_loss":kl_loss.item()})
                loss = model_loss + self.mu*kl_loss
                print("loss: ", loss.item())
            else:
                loss = model_loss
                print("loss: ", loss.item())

        gc.collect()
        torch.cuda.empty_cache()
        return (loss, outputs) if return_outputs else loss
