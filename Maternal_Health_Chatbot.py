# ---
# deploy: true
# ---

# # Serverless TensorRT-LLM (LLaMA 3 8B)

# In this example, we demonstrate how to use the TensorRT-LLM framework to serve Meta's LLaMA 3 8B model
# at very high throughput.

# We achieve a total throughput of over 25,000 output tokens per second on a single NVIDIA H100 GPU.
# At [Modal's on-demand rate](https://modal.com/pricing) of ~$4.50/hr, that's under $0.05 per million tokens --
# on auto-scaling infrastructure and served via a customizable API.

# Additional optimizations like speculative sampling can further improve throughput.

# ## Overview

# This guide is intended to document two things:
# the general process for building TensorRT-LLM on Modal
# and a specific configuration for serving the LLaMA 3 8B model.

# ### Build process

# Any given TensorRT-LLM service requires a multi-stage build process,
# starting from model weights and ending with a compiled engine.
# Because that process touches many sharp-edged high-performance components
# across the stack, it can easily go wrong in subtle and hard-to-debug ways
# that are idiosyncratic to specific systems.
# And debugging GPU workloads is expensive!

# This example builds an entire service from scratch, from downloading weight tensors
# to responding to requests, and so serves as living, interactive documentation of a TensorRT-LLM
# build process that works on Modal.

# ### Engine configuration

# TensorRT-LLM is the Lamborghini of inference engines: it achieves seriously
# impressive performance, but only if you tune it carefully.
# We carefully document the choices we made here and point to additional resources
# so you know where and how you might adjust the parameters for your use case.

# ## Installing TensorRT-LLM

# To run TensorRT-LLM, we must first install it. Easier said than done!

# In Modal, we define [container images](https://modal.com/docs/guide/custom-container) that run our serverless workloads.
# All Modal containers have access to GPU drivers via the underlying host environment,
# but we still need to install the software stack on top of the drivers, from the CUDA runtime up.

# We start from an official `nvidia/cuda` image,
# which includes the CUDA runtime & development libraries
# and the environment configuration necessary to run them.

from typing import Optional

import modal
import pydantic  # for typing, used later
import json  # needed for JSON serialization in streaming endpoint

tensorrt_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04",
    add_python="3.10",  # TRT-LLM requires Python 3.10
).entrypoint([])  # remove verbose logging by base image on entry

# On top of that, we add some system dependencies of TensorRT-LLM,
# including OpenMPI for distributed communication, some core software like `git`,
# and the `tensorrt_llm` package itself.

tensorrt_image = tensorrt_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
).pip_install(
    "tensorrt_llm==0.14.0",
    "pynvml<12",  # avoid breaking change to pynvml version API
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
)

# Note that we're doing this by [method-chaining](https://quanticdev.com/articles/method-chaining/)
# a number of calls to methods on the `modal.Image`. If you're familiar with
# Dockerfiles, you can think of this as a Pythonic interface to instructions like `RUN` and `CMD`.

# End-to-end, this step takes five minutes.
# If you're reading this from top to bottom,
# you might want to stop here and execute the example
# with `modal run trtllm_llama.py`
# so that it runs in the background while you read the rest.

# ## Downloading the Model

# Next, we download the model we want to serve. In this case, we're using the instruction-tuned
# version of Meta's LLaMA 3 8B model.
# We use the function below to download the model from the Hugging Face Hub.

MODEL_DIR = "/root/model/model_input"
MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"  # fork without repo gating
MODEL_REVISION = "b1532e4dee724d9ba63fe17496f298254d87ca64"  # pin model revisions to prevent unexpected changes!


def download_model():
    import os

    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
        revision=MODEL_REVISION,
    )
    move_cache()


# Just defining that function doesn't actually download the model, though.
# We can run it by adding it to the image's build process with `run_function`.
# The download process has its own dependencies, which we add here.

MINUTES = 60  # seconds
tensorrt_image = (  # update the image by downloading the model we're using
    tensorrt_image.pip_install(  # add utilities for downloading the model
        "hf-transfer==0.1.8",
        "huggingface_hub==0.26.2",
        "requests~=2.31.0",
    )
    .env(  # hf-transfer for faster downloads
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    )
    .run_function(  # download the model
        download_model,
        timeout=20 * MINUTES,
    )
)

# ## Quantization

# The amount of GPU RAM on a single card is a tight constraint for most LLMs:
# RAM is measured in billions of bytes and models have billions of parameters.
# The performance cliff if you need to spill to CPU memory is steep,
# so all of those parameters must fit in the GPU memory,
# along with other things like the KV cache.

# The simplest way to reduce LLM inference's RAM requirements is to make the model's parameters smaller,
# to fit their values in a smaller number of bits, like four or eight. This is known as _quantization_.

# We use a quantization script provided by the TensorRT-LLM team.
# This script takes a few minutes to run.

GIT_HASH = "b0880169d0fb8cd0363049d91aa548e58a41be07"
CONVERSION_SCRIPT_URL = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/{GIT_HASH}/examples/quantization/quantize.py"

# NVIDIA's Ada Lovelace/Hopper chips, like the 4090, L40S, and H100,
# are capable of native calculations in 8bit floating point numbers, so we choose that as our quantization format (`qformat`).
# These GPUs are capable of twice as many floating point operations per second in 8bit as in 16bit --
# about two quadrillion per second on an H100 SXM.

N_GPUS = 1  # Heads up: this example has not yet been tested with multiple GPUs
GPU_CONFIG = f"H100:{N_GPUS}"

DTYPE = "float16"  # format we download in, regular fp16
QFORMAT = "fp8"  # format we quantize the weights to
KV_CACHE_DTYPE = "fp8"  # format we quantize the KV cache to

# Quantization is lossy, but the impact on model quality can be minimized by
# tuning the quantization parameters based on target outputs.

CALIB_SIZE = "512"  # size of calibration dataset

# We put that all together with another invocation of `.run_commands`.

QUANTIZATION_ARGS = f"--dtype={DTYPE} --qformat={QFORMAT} --kv_cache_dtype={KV_CACHE_DTYPE} --calib_size={CALIB_SIZE}"

CKPT_DIR = "/root/model/model_ckpt"
tensorrt_image = (  # update the image by quantizing the model
    tensorrt_image.run_commands(  # takes ~2 minutes
        [
            f"wget {CONVERSION_SCRIPT_URL} -O /root/convert.py",
            f"python /root/convert.py --model_dir={MODEL_DIR} --output_dir={CKPT_DIR}"
            + f" --tp_size={N_GPUS}"
            + f" {QUANTIZATION_ARGS}",
        ],
        gpu=GPU_CONFIG,
    )
)

# ## Compiling the engine

# TensorRT-LLM achieves its high throughput primarily by compiling the model:
# making concrete choices of CUDA kernels to execute for each operation.
# These kernels are much more specific than `matrix_multiply` or `softmax` --
# they have names like `maxwell_scudnn_winograd_128x128_ldg1_ldg4_tile148t_nt`.
# They are optimized for the specific types and shapes of tensors that the model uses
# and for the specific hardware that the model runs on.

# That means we need to know all of that information a priori --
# more like the original TensorFlow, which defined static graphs, than like PyTorch,
# which builds up a graph of kernels dynamically at runtime.

# This extra layer of constraint on our LLM service is an important part of
# what allows TensorRT-LLM to achieve its high throughput.

# So we need to specify things like the maximum batch size and the lengths of inputs and outputs.
# The closer these are to the actual values we'll use in production, the better the throughput we'll get.

# Since we want to maximize the throughput, assuming we had a constant workload,
# we set the batch size to the largest value we can fit in GPU RAM.
# Quantization helps us again here, since it allows us to fit more tokens in the same RAM.

MAX_INPUT_LEN, MAX_OUTPUT_LEN = 256, 256
MAX_NUM_TOKENS = 2**17
MAX_BATCH_SIZE = (
    1024  # better throughput at larger batch sizes, limited by GPU RAM
)
ENGINE_DIR = "/root/model/model_output"

SIZE_ARGS = f"--max_input_len={MAX_INPUT_LEN} --max_num_tokens={MAX_NUM_TOKENS} --max_batch_size={MAX_BATCH_SIZE}"

# There are many additional options you can pass to `trtllm-build` to tune the engine for your specific workload.
# You can find the document we used for LLaMA
# [here](https://github.com/NVIDIA/TensorRT-LLM/tree/b0880169d0fb8cd0363049d91aa548e58a41be07/examples/llama),
# which you can use to adjust the arguments to fit your workloads,
# e.g. adjusting rotary embeddings and block sizes for longer contexts.
# We also recommend the [official TRT-LLM best practices guide](https://nvidia.github.io/TensorRT-LLM/performance/perf-best-practices.html).

# To make best use of our 8bit floating point hardware, and the weights and KV cache we have quantized,
# we activate the 8bit floating point fused multi-head attention plugin.

# Because we are targeting maximum throughput, we do not activate the low latency 8bit floating point matrix multiplication plugin
# or the 8bit floating point matrix multiplication (`gemm`) plugin, which documentation indicates target smaller batch sizes.

PLUGIN_ARGS = "--use_fp8_context_fmha enable"


tensorrt_image = (  # update the image by building the TensorRT engine
    tensorrt_image.run_commands(  # takes ~5 minutes
        [
            f"trtllm-build --checkpoint_dir {CKPT_DIR} --output_dir {ENGINE_DIR}"
            + f" --workers={N_GPUS}"
            + f" {SIZE_ARGS}"
            + f" {PLUGIN_ARGS}"
        ],
        gpu=GPU_CONFIG,  # TRT-LLM compilation is GPU-specific, so make sure this matches production!
    ).env(  # show more log information from the inference engine
        {"TLLM_LOG_LEVEL": "INFO"}
    )
)

app = modal.App(
    f"maternal-health-chatbot-{MODEL_ID.split('/')[-1]}", image=tensorrt_image
)


@app.cls(
    gpu=GPU_CONFIG,
    image=tensorrt_image,
)
class Model:
    @modal.enter()
    def load(self):
        """Loads the TRT-LLM engine and configures our tokenizer.

        The @enter decorator ensures that it runs only once per container, when it starts."""
        import time

        print(
            f"{COLOR['HEADER']}🥶 Cold boot: spinning up TRT-LLM engine{COLOR['ENDC']}"
        )
        self.init_start = time.monotonic_ns()

        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # LLaMA models do not have a padding token, so we use the EOS token
        self.tokenizer.add_special_tokens(
            {"pad_token": self.tokenizer.eos_token}
        )
        # and then we add it from the left, to minimize impact on the output
        self.tokenizer.padding_side = "left"
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id

        runner_kwargs = dict(
            engine_dir=f"{ENGINE_DIR}",
            lora_dir=None,
            rank=tensorrt_llm.mpi_rank(),  # this will need to be adjusted to use multiple GPUs
            max_output_len=MAX_OUTPUT_LEN,
        )

        self.model = ModelRunner.from_dir(**runner_kwargs)

        self.init_duration_s = (time.monotonic_ns() - self.init_start) / 1e9
        print(
            f"{COLOR['HEADER']}🚀 Cold boot finished in {self.init_duration_s}s{COLOR['ENDC']}"
        )
        
        # Define the system prompt for maternal health
        self.maternal_health_system_prompt = """You are a maternal health expert providing evidence-based information to expectant and new mothers. Your goal is to:

1. Provide accurate medical information based on current best practices in obstetrics and maternal health
2. Offer practical advice for common pregnancy and postpartum concerns
3. Recognize warning signs that require immediate medical attention
4. Support maternal mental health by acknowledging emotional challenges
5. Respect cultural differences in maternal care practices while promoting safety
6. Always prioritize the mother and baby's health and safety in your recommendations
7. Be clear about when someone should seek professional medical advice

Important disclaimers:
- You are NOT providing personalized medical diagnosis or replacing healthcare providers
- For any emergency situations, ALWAYS advise seeking immediate medical attention
- When discussing medications, emphasize the importance of consulting healthcare providers
- Be clear about the limitations of your knowledge and current medical consensus

When responding to questions, structure your answers to:
1. Acknowledge the concern with empathy
2. Provide evidence-based information
3. Offer practical suggestions when appropriate
4. Flag any warning signs that require medical attention
5. Recommend professional consultation when needed
"""

    @modal.method()
    def generate(self, question: str, settings=None):
        """Generate a response to a single prompt, optionally with custom inference settings."""
        import time

        if settings is None or not settings:
            settings = dict(
                temperature=0.3,  # Slightly increased for more natural medical responses
                top_k=40,
                stop_words_list=None,
                repetition_penalty=1.1,
            )

        settings["max_new_tokens"] = (
            MAX_OUTPUT_LEN  # exceeding this will raise an error
        )
        settings["end_id"] = self.end_id
        settings["pad_id"] = self.pad_id

        print(
            f"{COLOR['HEADER']}🚀 Generating maternal health response...{COLOR['ENDC']}"
        )
        start = time.monotonic_ns()

        # Apply the maternal health system prompt
        parsed_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.maternal_health_system_prompt},
                {"role": "user", "content": question}
            ],
            add_generation_prompt=True,
            tokenize=False,
        )

        print(
            f"{COLOR['HEADER']}Parsed prompt:{COLOR['ENDC']}",
            parsed_prompt,
            sep="\n\t",
        )

        inputs_t = self.tokenizer(
            [parsed_prompt], return_tensors="pt", padding=True, truncation=False
        )["input_ids"]

        print(
            f"{COLOR['HEADER']}Input tensor:{COLOR['ENDC']}", inputs_t[:, :8]
        )

        outputs_t = self.model.generate(inputs_t, **settings)

        output_text = self.tokenizer.batch_decode(
            outputs_t[:, 0]
        )[0]  # only one output, so we index with 0

        response = extract_assistant_response(output_text)
        
        duration_s = (time.monotonic_ns() - start) / 1e9
        num_tokens = len(self.tokenizer.encode(response))

        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}{question}",
            f"\n{COLOR['BLUE']}{response}",
            "\n\n",
            sep=COLOR["ENDC"],
        )

        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {MODEL_ID} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.{COLOR['ENDC']}"
        )

        return response
    
    @modal.method()
    def generate_stream(self, question: str, settings=None):
        """Generate a streaming response, yielding chunks as they're generated."""
        import time
        import threading
        import queue

        if settings is None or not settings:
            settings = dict(
                temperature=0.3,
                top_k=40,
                stop_words_list=None,
                repetition_penalty=1.1,
            )

        settings["max_new_tokens"] = MAX_OUTPUT_LEN
        settings["end_id"] = self.end_id
        settings["pad_id"] = self.pad_id
        settings["streaming"] = True  # Enable streaming mode

        print(f"{COLOR['HEADER']}🚀 Starting streaming generation...{COLOR['ENDC']}")
        start = time.monotonic_ns()

        # Apply the maternal health system prompt
        parsed_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.maternal_health_system_prompt},
                {"role": "user", "content": question}
            ],
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs_t = self.tokenizer(
            [parsed_prompt], return_tensors="pt", padding=True, truncation=False
        )["input_ids"]

        # Simulate streaming with a simple approach
        # In a real application, you might want to use a more sophisticated approach
        text_queue = queue.Queue()
        full_response = ""
        
        def process_generation():
            nonlocal full_response
            outputs_t = self.model.generate(inputs_t, **settings)
            output_text = self.tokenizer.batch_decode(outputs_t[:, 0])[0]
            response = extract_assistant_response(output_text)
            full_response = response
            
            # Split the response into chunks and put them in the queue
            chunks = [response[i:i+10] for i in range(0, len(response), 10)]
            for chunk in chunks:
                text_queue.put(chunk)
            text_queue.put(None)  # Signal end of generation
        
        # Start generation in a separate thread
        thread = threading.Thread(target=process_generation)
        thread.start()
        
        # Yield chunks as they become available
        while True:
            chunk = text_queue.get()
            if chunk is None:
                break
            yield chunk
            time.sleep(0.1)  # Small delay to simulate streaming
        
        duration_s = (time.monotonic_ns() - start) / 1e9
        num_tokens = len(self.tokenizer.encode(full_response))
        
        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Streaming completed: Generated {num_tokens} tokens in {duration_s:.1f} seconds{COLOR['ENDC']}"
        )

@app.local_entrypoint()
def main():
    # Single maternal health question for testing
    question = "Is morning sickness normal?"
    
    model = Model()
    response = model.generate.remote(question)
    print(f"Question: {question}")
    print(f"Response: {response}")


web_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "fastapi[standard]==0.115.4",
    "pydantic==2.9.2",
    "starlette==0.41.2",
)


class MaternalHealthRequest(pydantic.BaseModel):
    question: str
    settings: Optional[dict] = None


@app.function(image=web_image)
@modal.web_endpoint(method="POST")
def maternal_health_api(data: MaternalHealthRequest) -> dict:
    """Generate a response to a maternal health question.
    
    This endpoint accepts a single question related to maternal health and returns an evidence-based response.
    """
    response = Model.generate.remote(data.question, settings=data.settings)
    return {
        "question": data.question,
        "response": response
    }


@app.function(image=web_image)
@modal.web_endpoint(method="POST")
async def maternal_health_stream(data: MaternalHealthRequest):
    """Stream a response to a maternal health question.
    
    This endpoint accepts a single question and streams the response chunk by chunk.
    """
    from fastapi.responses import StreamingResponse
    
    async def stream_response():
        generator = Model.generate_stream.remote_gen(data.question, settings=data.settings)
        for chunk in generator:
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )

# To deploy this API, run: modal deploy Maternal_Health_Chatbot.py
# To test it locally, run: modal serve Maternal_Health_Chatbot.py
# Access the API documentation at: https://your-modal-url/docs

# ## Footer

# The rest of the code in this example is utility code.


COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}


def extract_assistant_response(output_text):
    """Model-specific code to extract model responses.

    See this doc for LLaMA 3: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/."""
    # Split the output text by the assistant header token
    parts = output_text.split("<|start_header_id|>assistant<|end_header_id|>")

    if len(parts) > 1:
        # Join the parts after the first occurrence of the assistant header token
        response = parts[1].split("<|eot_id|>")[0].strip()

        # Remove any remaining special tokens and whitespace
        response = response.replace("<|eot_id|>", "").strip()

        return response
    else:
        return output_text