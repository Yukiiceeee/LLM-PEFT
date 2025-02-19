from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./final_saved_models"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

from transformers import pipeline

text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, num_return_sequences=1)

prompt = "hello! tell me who are you?"
outputs = text_generator(prompt, max_new_tokens=100)

print("输出结构：", outputs)

generated_text = outputs[0]["generated_text"]
print("生成的文本：", generated_text)