import os

from typing import Dict, Optional, List
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath, PromptAdapterPath
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.logger import RequestLogger

logger = logging.getLogger("ray.serve")

app = FastAPI()

from transformers import AutoTokenizer

def get_chat_template(model_name: str) -> str:
    from pathlib import Path
    import json
    if os.path.isdir(model_name):
        path = Path(model_name)
    else:
        path = model_name

    chat_template = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=os.path.isdir(model_name))
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            logger.info("✅ Loaded chat_template from tokenizer.")
        else:
            raise ValueError("chat_template is None")
    except Exception as e:
        logger.error(f"⚠️ Tokenizer loading failed or chat_template missing: {e}")
        # Try fallback: manually load tokenizer_config.json
        config_path = Path(path) / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                chat_template = config.get("chat_template")
                if chat_template:
                    logger.info("✅ Loaded chat_template from tokenizer_config.json.")
                else:
                    logger.error("❌ chat_template not found in tokenizer_config.json. Loading the default chat template")
                    chat_template = """
        {{- bos_token }}
    {%- if custom_tools is defined %}
        {%- set tools = custom_tools %}
    {%- endif %}
    {%- if not tools_in_user_message is defined %}
        {%- set tools_in_user_message = true %}
    {%- endif %}
    {%- if not date_string is defined %}
        {%- if strftime_now is defined %}
            {%- set date_string = strftime_now("%d %b %Y") %}
        {%- else %}
            {%- set date_string = "26 Jul 2024" %}
        {%- endif %}
    {%- endif %}
    {%- if not tools is defined %}
        {%- set tools = none %}
    {%- endif %}

    {#- This block extracts the system message, so we can slot it into the right place. #}
    {%- if messages[0]['role'] == 'system' %}
        {%- set system_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {%- set system_message = "" %}
    {%- endif %}

    {#- System message #}
    {{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
    {%- if tools is not none %}
        {{- "Environment: ipython\n" }}
    {%- endif %}
    {{- "Cutting Knowledge Date: December 2023\n" }}
    {{- "Today Date: " + date_string + "\n\n" }}
    {%- if tools is not none and not tools_in_user_message %}
        {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
        {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
        {{- "Do not use variables.\n\n" }}
        {%- for t in tools %}
            {{- t | tojson(indent=4) }}
            {{- "\n\n" }}
        {%- endfor %}
    {%- endif %}
    {{- system_message }}
    {{- "<|eot_id|>" }}

    {#- Custom tools are passed in a user message with some extra guidance #}
    {%- if tools_in_user_message and not tools is none %}
        {#- Extract the first user message so we can plug it in here #}
        {%- if messages | length != 0 %}
            {%- set first_user_message = messages[0]['content']|trim %}
            {%- set messages = messages[1:] %}
        {%- else %}
            {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
    {%- endif %}
        {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
        {{- "Given the following functions, please respond with a JSON for a function call " }}
        {{- "with its proper arguments that best answers the given prompt.\n\n" }}
        {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
        {{- "Do not use variables.\n\n" }}
        {%- for t in tools %}
            {{- t | tojson(indent=4) }}
            {{- "\n\n" }}
        {%- endfor %}
        {{- first_user_message + "<|eot_id|>"}}
    {%- endif %}

    {%- for message in messages %}
        {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
            {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
        {%- elif 'tool_calls' in message %}
            {%- if not message.tool_calls|length == 1 %}
                {{- raise_exception("This model only supports single tool-calls at once!") }}
            {%- endif %}
            {%- set tool_call = message.tool_calls[0].function %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
            {{- "<|eot_id|>" }}
        {%- elif message.role == "tool" or message.role == "ipython" %}
            {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
            {%- if message.content is mapping or message.content is iterable %}
                {{- message.content | tojson }}
            {%- else %}
                {{- message.content }}
            {%- endif %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
    {%- endif %}
    """
    
        else:
            logger.error("❌ tokenizer_config.json not found at:", config_path)
       
    return chat_template


deployment_name = os.environ.get("DEPLOYMENT_NAME", "VLLMDeployment")

@serve.deployment(name=deployment_name)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            # Determine the name of the served model for the OpenAI client.
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                served_model_names,
                self.response_role,
                lora_modules=self.lora_modules,
                prompt_adapters=self.prompt_adapters,
                request_logger=self.request_logger,
                chat_template=self.chat_template,
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    @app.get("/v1/model-info")
    async def model_info(self):
        return {"model": self.engine_args.model}
    

def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )

    parser = make_arg_parser(arg_parser)
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True
    
    chat_template = cli_args.get("chat_template")
    if chat_template is None:
        logger.info("Using Default Template")
        print("Using Default Template")

        chat_template = get_chat_template(engine_args.model)

    return VLLMDeployment.bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.prompt_adapters,
        cli_args.get("request_logger"),
        chat_template,
    )


model = build_app(
    {
        "model": os.environ['MODEL_ID'], 
        "served_model_name" : os.environ.get("SERVED_MODEL_NAME", None),
        "tensor-parallel-size": os.environ.get("TENSOR_PARALLELISM",1), 
        "pipeline-parallel-size": os.environ.get("PIPELINE_PARALLELISM",1),
        "max-model-len": os.environ.get("MAX_MODEL_LEN", "2048"),
        "quantization": os.environ.get("QUANTIZATION", "fp16"),
        "chat_template" : os.environ.get("CHAT_TEMPLATE", None),
    })
