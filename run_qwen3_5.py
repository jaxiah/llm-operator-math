from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen3_5ForConditionalGeneration.from_pretrained("Qwen/Qwen3.5-2B", torch_dtype="auto", device_map="auto")

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "./demo.jpeg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]


# Preparation for inference
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text)
