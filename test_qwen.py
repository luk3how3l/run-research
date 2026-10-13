from transformers import AutoModelForVision2Seq, Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"


print("Loading model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",
)

processor = AutoProcessor.from_pretrained(MODEL)

print("Model loaded.")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "test_bib.jpg",
            },
            {
                "type": "text",
                "text": """
You are reading a race bib.

Identify ONLY the bib number.

Ignore:
- jersey numbers
- sponsor logos
- text
- lane numbers
- other numbers in the image

If you can clearly read the bib number, return ONLY the number.

If you cannot determine it confidently, return:
UNKNOWN
""",
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = inputs.to("cpu")

print("Running inference...")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=10,
    )

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(
        inputs["input_ids"],
        generated_ids
    )
]

output = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("\nQwen result:")
print(output[0])