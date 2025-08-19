# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# !pip install "peft==0.2.0"
# !pip install "transformers==4.27.2" "datasets==2.9.0" "accelerate==0.17.1" "evaluate==0.4.0" "bitsandbytes==0.37.1" loralib --upgrade --quiet

#pip install peft transformers datasets accelerate evaluate bitsandbytes loralib --upgrade --quiet
# install additional dependencies needed for training
# !pip install rouge-score tensorboard py7zr

# # %%
# !pip install peft transformers datasets accelerate evaluate bitsandbytes loralib --upgrade --quiet

# # %%
# !pip install --upgrade torch torchvision torchaudio

# # %%
# !pip install tensorboard matplotlib

# %%
from datasets import load_from_disk, concatenate_datasets, DatasetDict

#textile patent documents
dataset = DatasetDict({
    'train': load_from_disk('../Textile_Patent_(70-20-10)_Aug_LexRank_thres_3/train'),
    'validation': load_from_disk('../Textile_Patent_(70-20-10)_Aug_LexRank_thres_3/validation'),
    'test': load_from_disk('../Textile_Patent_(70-20-10)_Aug_LexRank_thres_3/test')
})
#dataset = load_from_disk('/content/drive/MyDrive/FYP-Abstractive Text Summarization/Data/Textile_Patent_(70-20-10)_LexRank')

# # %%
# print(f"Train dataset size: {len(dataset['train'])}")
# print(f"Validation dataset size: {len(dataset['validation'])}")
# print(f"Test dataset size: {len(dataset['test'])}")

# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id="facebook/bart-large"

# Load tokenizer of bart-large
tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# %%
#len(dataset["train"]['description'][4])
# min_len = 9999999
# for i in range(len(dataset['train'])):
#   temp = len(dataset["train"]['description'][i])
#   if (temp < min_len):
#     min_len = temp

# print(min_len)

# %%
# from datasets import concatenate_datasets
# import numpy as np
# # The maximum total input sequence length after tokenization.
# # Sequences longer than this will be truncated, sequences shorter will be padded.
# tokenized_inputs = dataset["train"].map(lambda x: tokenizer(x["description"], truncation=True), batched=True, remove_columns=["description", "abstract"])
# input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# # take 85 percentile of max length for better utilization
# max_source_length = int(np.percentile(input_lenghts, 85))
# print(f"Max source length: {max_source_length}")

# # The maximum total sequence length for target text after tokenization.
# # Sequences longer than this will be truncated, sequences shorter will be padded."
# tokenized_targets = dataset["train"].map(lambda x: tokenizer(x["abstract"], truncation=True), batched=True, remove_columns=["description", "abstract"])
# target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# # take 90 percentile of max length for better utilization
# max_target_length = int(np.percentile(target_lenghts, 90))
# print(f"Max target length: {max_target_length}")

# %%
#Maximum available for bart model
max_source_length = 1024
max_target_length = 512

# %%
def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for bart
    #inputs = ["summarize: " + extract_important_sentences(item) for item in sample["description"]]
    inputs = ["summarize: " + item for item in sample["description"]]


    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["abstract"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# %%
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["description", "abstract"])
# print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# %%
# # save datasets to disk for later easy loading
# tokenized_dataset["train"].save_to_disk("../Adapter_Tuning/LoRA+Bart/tokenized_data_S2_EXT4_M2L/train")
# tokenized_dataset["validation"].save_to_disk("../Adapter_Tuning/LoRA+Bart/tokenized_data_S2_EXT4_M2L/validation")
# tokenized_dataset["test"].save_to_disk("../Adapter_Tuning/LoRA+Bart/tokenized_data_S2_EXT4_M2L/test")

# %%
# # Load datasets
# tokenized_dataset = DatasetDict({
#     'train': load_from_disk('/content/drive/MyDrive/FYP-Abstractive Text Summarization/Adapter_Tuning/LoRA+Bart/tokenized_data_S2_EXT3/train'),
#     'validation': load_from_disk('/content/drive/MyDrive/FYP-Abstractive Text Summarization/Adapter_Tuning/LoRA+Bart/tokenized_data_S2_EXT3/validation'),
#     'test': load_from_disk('/content/drive/MyDrive/FYP-Abstractive Text Summarization/Adapter_Tuning/LoRA+Bart/tokenized_data_S2_EXT3/test')
# })

# %%
# tokenized_data = {
#     'train': train,
#     'validation': valid,
#     'test': test
# }

# %%
# tokenized_dataset

# %%
from transformers import AutoModelForSeq2SeqLM

# huggingface hub model id
model_id = "facebook/bart-large"

# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# %%
# !pip list | grep cuda

# %%
# print(model)

# %%
# import torch
# #check for gpu
# if torch.backends.mps.is_available():
#    mps_device = torch.device("mps")
#    x = torch.ones(1, device=mps_device)
#    print (x)
# else:
#    print ("MPS device not found.")

# %%
# import platform
# print(platform.platform())

# %%
from peft import LoraConfig
from peft import get_peft_model
from peft import prepare_model_for_kbit_training
from peft import TaskType
# Define LoRA Config
lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q_proj", "k_proj", "v_proj"],
 lora_dropout=0.05,
 bias="none",
 init_lora_weights="gaussian",
 task_type=TaskType.SEQ_2_SEQ_LM
)
# prepare int-8 model for training
model = prepare_model_for_kbit_training(model)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %%
from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

# %%
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback

output_dir="../Adapter_Tuning/LoRA+Bart/M2L_LR_S2_EXT4_EXP17"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    learning_rate=1e-4, 
    num_train_epochs=100,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="epoch",
    evaluation_strategy = "epoch",
    save_total_limit = 10,
    #logging_steps=500,
    # eval_steps=500,
    save_strategy="epoch",
    report_to="tensorboard",
    load_best_model_at_end = True,
    metric_for_best_model = 'loss'
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    callbacks = [EarlyStoppingCallback(early_stopping_patience=7)]
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


# %%
# device = torch.device("mps:0")
# model.to(device)

# %%
# print(model)

# %%
# train model
trainer.train()

# %%
# Save our LoRA model & tokenizer results
peft_model_id="M2L_LR_S2_EXT3_EXP17"
trainer.model.save_pretrained(f'../Adapter_Tuning/LoRA+Bart/{peft_model_id}_model')
tokenizer.save_pretrained(f'../Adapter_Tuning/LoRA+Bart/{peft_model_id}_tokenizer')
# if you want to save the base model to call
# trainer.model.base_model.save_pretrained(peft_model_id)

# %%
history = trainer.state.log_history[:len(trainer.state.log_history)-1]
training_loss = []
eval_loss = []
for i in range(len(history)):
    if i%2 == 0:
        training_loss.append(history[i]['loss'])
    else:
        eval_loss.append(history[i]['eval_loss'])

# %%
epochs = range(1, len(training_loss) + 1)

# %%
import matplotlib.pyplot as plt
# Plot training and validation loss
plt.plot(epochs, training_loss, label='Training Loss')
plt.plot(epochs, eval_loss, label='Validation Loss')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Show plot
plt.savefig('Loss_Curve_EXP16.png')

# %%
# test_dataset = load_from_disk("/content/drive/MyDrive/FYP-Abstractive Text Summarization/Adapter_Tuning/LoRA+T5/tokenized_data/test").with_format("torch")

# # %%
# for sample in test_dataset:
#  input = tokenizer.decode(sample['input_ids'].detach().cpu().numpy(), skip_special_tokens=True)
#  output = model.generate(input_ids=sample["input_ids"].unsqueeze(0).cuda(), do_sample=True, top_p=0.9, max_new_tokens=max_target_length)
#  prediction = tokenizer.decode(output[0].detach().cpu().numpy(), skip_special_tokens=True)
#  break

# # %%
# input

# # %%
# prediction

# %%



