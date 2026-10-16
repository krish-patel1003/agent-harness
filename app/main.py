import argparse
import os
import sys
import json
from collections.abc import Callable
from typing import Any

from openai import OpenAI

class ToolRegistry:

    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}
        self._functions: dict[str, Callable[[dict[str, Any]], str]] = {}

    def register_tool(
            self, 
            name: str, 
            description: str, 
            parameters: dict[str, Any],
            function: Callable[[dict[str, Any]], str]) -> None:

        self._tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        self._functions[name] = function

    def definition(self) -> list[dict[str, Any]]:
        return list(self._tools.values())

    def get_function(self, name: str) -> Callable[[dict[str, Any]], str] | None:
        return self._functions.get(name, None)

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        function = self.get_function(name)
        if function is None:
            raise ValueError(f"Tool '{name}' is not registered.")
        return function(arguments)


def read_file(args: dict[str, Any]) -> str:
    file_path = args.get("file_path")
    if not file_path:
        raise ValueError("Missing 'file_path' argument.")
    with open(file_path, 'r') as f:
        return f.read()


def write_file(args: dict[str, Any]) -> str:
    file_path = args.get("file_path")
    content = args.get("content")
    if not file_path or content is None:
        raise ValueError("Missing 'file_path' or 'content' argument.")
    with open(file_path, 'w') as f:
        f.write(content)
    return f"Content written to {file_path}"


def execute_bash(args: dict[str, Any]) -> str:
    command = args.get("command")
    # return the stdout or the stderr of the command
    if not command:
        raise ValueError("Missing 'command' argument.")
    import subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout or result.stderr


class Agent:

    def __init__(
            self,
            client: OpenAI,
            tools: ToolRegistry,
            model: str,
            max_turns: int = 20,
            ) -> None:
        self.client = client
        self.tools = tools
        self.model = model
        self.max_turns = max_turns

    def run(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        for turn in range(self.max_turns):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools.definition()
            )

            if not response.choices or len(response.choices) == 0:
                raise RuntimeError("No choices in response")

            message = response.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                return message.content or ""

            if message.tool_calls:
                for call in message.tool_calls:
                    tool_name = call.function.name
                    arguments = json.loads(call.function.arguments)
                    result = self.tools.execute(tool_name, arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result
                    })

        raise RuntimeError("Max turns exceeded without a final response.")
    
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    API_KEY = os.getenv("OPENROUTER_API_KEY")
    BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    tools = ToolRegistry()
    tools.register_tool(
        name="Read",
        description="Read and return the contents of a file",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to read"
                }
            },
            "required": ["file_path"]
        },
        function=read_file
    )
    tools.register_tool(
        name="Write",
        description="Write content to a file",
        parameters={
            "type": "function",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        },
        function=write_file
    )

    tools.register_tool(
        name="Bash",
        description="Execute a bash command and return the output",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                }
            },
            "required": ["command"]
        },
        function=execute_bash
    )
    agent = Agent(client=client, tools=tools, model="anthropic/claude-haiku-4.5")

    print("Logs from your program will appear here!", file=sys.stderr)
    print(agent.run(args.p))


if __name__ == "__main__":
    main()
