from transformers import AutoModelForCausalLM, AutoTokenizer
from data_prepare import samples
import json
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer, TrainingArguments

model_name = "/d2/mxy/Models/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

with open("./dataset/dataset.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        json_line = json.dumps(s, ensure_ascii=False)
        f.write(json_line + "\n")
    else:
        print("data prepare done")

dataset = load_dataset("json", data_files="./dataset/dataset.jsonl", split="train")

train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

def tokenize_function(examples):
    texts = [f"{prompt}\n{completion}" for prompt, completion in zip(examples["prompt"], examples["completion"])]
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=10,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    run_name="deepseek-r1-distill-qwen-1.5b-lora",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
)

print("Training...")
trainer.train()
print("Saving model...")
save_path = "./saved_models"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

from peft import PeftModel
final_save_path = "./final_saved_models"
base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, save_path)
model = model.merge_and_unload()
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)

print("Done!")