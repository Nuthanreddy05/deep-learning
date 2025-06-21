from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

# Load dataset
dataset = load_dataset("beans", split={"train":"train[:80%]", "validation":"train[80%:]"})

# Labels
labels = dataset["train"].features["labels"].names
id2label = {i:label for i,label in enumerate(labels)}
label2id = {v:k for k,v in id2label.items()}

# Preprocessor and model
checkpoint = "google/vit-base-patch16-224-in21k"
processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForImageClassification.from_pretrained(
    checkpoint, num_labels=len(labels), id2label=id2label, label2id=label2id
)

# Preprocessing function
def preprocess(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    inputs = processor(images, return_tensors="pt")
    inputs["labels"] = batch["labels"]
    return inputs

# Apply preprocessing
dataset = dataset.map(preprocess, batched=True)

# Metric
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=preds, references=eval_pred.label_ids)

# Training setup
args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    remove_unused_columns=False,
)
trainer = Trainer(model, args, train_dataset=dataset["train"],
                  eval_dataset=dataset["validation"], compute_metrics=compute_metrics)

# Train & save
trainer.train()
model.save_pretrained("model")
processor.save_pretrained("model")
