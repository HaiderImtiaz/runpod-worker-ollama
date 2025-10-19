import runpod
import base64
import os
from utils import JobInput
from engine import OllamaEngine, OllamaOpenAiEngine

DEFAULT_MAX_CONCURRENCY = 8
max_concurrency = int(os.getenv("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))

async def handler(job):
    """
    Handles a RunPod job.
    Expected input:
    {
        "prompt": "Describe what’s in this image",
        "image": "<base64 encoded image data>",
        "model": "llama3.2-vision:11b"
    }
    """
    print("Incoming job:", job)

    job_input = job.get("input", {})
    prompt = job_input.get("prompt", "")
    image_b64 = job_input.get("image", None)
    model = job_input.get("model", "llama3.2-vision:11b")

    # Decode and save the image if present
    image_path = None
    if image_b64:
        image_data = base64.b64decode(image_b64)
        image_path = "/tmp/uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(image_data)
        print(f"Saved image to {image_path}")

    # Initialize the Ollama engine (OpenAI-compatible version)
    engine_class = OllamaOpenAiEngine if job_input.get("openai_route", False) else OllamaEngine
    engine = engine_class(model=model)

    # Create a JobInput-like structure for generate()
    task_input = JobInput({
        "prompt": prompt,
        "model": model,
        "image": image_path,
    })

    results = engine.generate(task_input)

    # Stream or yield batches of text
    async for batch in results:
        yield batch

# Start the RunPod serverless handler
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda _: max_concurrency,
    "return_aggregate_stream": True,
})
