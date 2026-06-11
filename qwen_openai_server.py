import os
import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "700"))

app = FastAPI(title="InfraGraph Qwen OpenAI-Compatible Server")

print(f"Loading model: {MODEL_ID}")
print(f"Torch: {torch.__version__}")
print(f"CUDA/HIP available via torch.cuda: {torch.cuda.is_available()}")
print(f"HIP version: {getattr(torch.version, 'hip', None)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
)

if device == "cpu":
    model = model.to("cpu")

model.eval()
print(f"Model loaded on: {device}")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = 0.1
    max_tokens: int | None = 700


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "infragraph-ai",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        prompt += "\nASSISTANT:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens or MAX_NEW_TOKENS,
            do_sample=(req.temperature or 0) > 0,
            temperature=req.temperature or 0.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return {
        "id": f"chatcmpl-infragraph-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
    }
