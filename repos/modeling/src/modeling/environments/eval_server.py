import asyncio
import gc
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class InferenceRequest(BaseModel):
    messages: list[dict[str, Any]]


class ModelChangeRequest(BaseModel):
    model_name: str
    temperature: float | None = 0.7
    top_p: float | None = 0.8


class ModelServer:
    def __init__(self, model_name: str, temperature: float = 0.7, top_p: float = 0.8):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.hf_model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        print(f"Loading model: {self.model_name}")

        self.hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        print(f"Model {self.model_name} loaded!")

    def cleanup(self):
        """Clean up model from memory"""
        if self.hf_model:
            del self.hf_model
        if self.processor:
            del self.processor
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Model {self.model_name} cleaned up")

    def inference(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.hf_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text


class InferenceQueue:
    def __init__(self):
        self.model_server = None
        self.queue = asyncio.Queue()
        self.worker_task = None

    def start(self):
        """Start the worker task"""
        self.worker_task = asyncio.create_task(self._worker())

    def update_model(self, model_server):
        """Update the model server"""
        if self.model_server:
            self.model_server.cleanup()
        self.model_server = model_server

    async def generate(self, messages):
        """Submit a request and wait for result"""
        if not self.model_server:
            raise HTTPException(
                status_code=503,
                detail="No model loaded. Use /change_model to load a model first.",
            )

        future = asyncio.Future()
        await self.queue.put((messages, future))
        return await future

    async def _worker(self):
        """Process requests one by one"""
        while True:
            try:
                messages, future = await self.queue.get()

                if not self.model_server:
                    future.set_exception(Exception("No model loaded"))
                    self.queue.task_done()
                    continue

                try:
                    result = self.model_server.inference(messages)
                    future.set_result(result[0] if result else "")
                except Exception as e:
                    future.set_exception(e)

                self.queue.task_done()

            except Exception as e:
                print(f"Worker error: {e}")


app = FastAPI()
inference_queue = None


@app.on_event("startup")
async def startup():
    global inference_queue
    inference_queue = InferenceQueue()
    inference_queue.start()
    print("Server started - no model loaded yet. Use /change_model to load a model.")


@app.post("/generate")
async def generate(request: InferenceRequest):
    try:
        result = await inference_queue.generate(request.messages)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/change_model")
async def change_model(request: ModelChangeRequest):
    try:
        print(f"Changing model to: {request.model_name}")

        # Create new model server
        new_model = ModelServer(request.model_name, request.temperature, request.top_p)

        # Update queue with new model (this will cleanup old model)
        inference_queue.update_model(new_model)

        return {
            "status": "success",
            "message": f"Model changed to {request.model_name}",
            "model_name": request.model_name,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load model: {e!s}"
        ) from e


@app.get("/model_info")
async def model_info():
    if not inference_queue or not inference_queue.model_server:
        return {"status": "no_model_loaded"}

    return {
        "model_name": inference_queue.model_server.model_name,
        "temperature": inference_queue.model_server.temperature,
        "top_p": inference_queue.model_server.top_p,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": inference_queue and inference_queue.model_server is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
