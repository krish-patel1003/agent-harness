import argparse
import os
import sys
import json
from collections.abc import Callable
from typing import Any
from pathlib import Path
from html import escape
import re

from openai import OpenAI

class Skills:
    """
    Class to manage and skills 
    """

    

    def __init__(self):
        self.skills: dict[str, dict[str, Any]] = {}

    @staticmethod
    def load_skills_catalog():
        skills = []
        skills_dir = Path.cwd() / ".claude" / "skills"

        for path in sorted(skills_dir.glob("*/SKILL.md")):
            content = path.read_text(encoding="utf-8")
            match = re.match(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", content, re.DOTALL)

            if not match:
                raise ValueError(f"Skill file {path} is missing frontmatter")

            frontmatter = match.group(1)
            fields = dict(
                re.findall(r"^(name|description|context):[ \t]*(.+?)[ \t]*$", frontmatter, re.MULTILINE)
            )

            if "name" not in fields or "description" not in fields:
                raise ValueError(f"Skill file {path} is missing name or description")

            skills.append({
                "name": fields["name"].strip("\"'"),
                "description": fields["description"].strip("\"'"),
                "context": fields.get("context", "").strip("\"'"),
                "path": str(path)
            })

        return skills

    @staticmethod
    def skills_prompt(skills: list[dict[str, Any]]) -> str:
        prompt = (
            "You have access to the following skills. When a skill matches the task, "
            "call the Skill tool with its name and arguments, then follow the returned "
            "instructions using the available tools. You can load multiple skills. "
            "For context: fork skills, the tool runs a subagent and returns its completed "
            "result; use that result without repeating the subagent's work. "
            "Skills already expanded in the conversation do not need to be loaded again.\n\n"
        )
        for skill in skills:
            prompt += f"- {skill['name']}: {skill['description']} (path: {skill['path']})\n"
        return prompt

    @staticmethod
    def invoke(
        args: dict[str, Any],
        skills: list[dict[str, Any]],
        execute: Callable[[dict[str, Any], str], str],
    ) -> str:
        name = args.get("skill")
        arguments = args.get("args", "")
        if not isinstance(name, str) or not name or any(char.isspace() for char in name):
            return "Error: 'skill' must be a single nonempty skill name."
        if not isinstance(arguments, str):
            return "Error: 'args' must be a string."

        command = name if name.startswith("/") else f"/{name}"
        try:
            return execute(Skills.find(command[1:], skills), arguments)
        except (ValueError, OSError, RuntimeError) as error:
            return f"Error loading skill: {error}"

    @staticmethod
    def expand_slash_command(prompt: str, skills: list[dict[str, Any]]) -> str:
        if not prompt.lstrip().startswith("/"):
            return prompt

        parts = prompt.lstrip().split(maxsplit=1)
        command = parts[0][1:]
        arguments = parts[1] if len(parts) > 1 else ""

        skill = Skills.find(command, skills)
        instructions = Skills.render(skill, arguments)
        return (
            f"Run the /{command} skill using these instructions.\n"
            f"Resolve relative file references from: {Path(skill['path']).parent}\n\n"
            f"{instructions}"
        )

    @staticmethod
    def find(command: str, skills: list[dict[str, Any]]) -> dict[str, Any]:
        matches = [
            skill for skill in skills
            if command in (skill["name"], Path(skill["path"]).parent.name)
        ]
        if not matches:
            raise ValueError(f"Unknown skill command: /{command}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous skill command: /{command}")

        return matches[0]

    @staticmethod
    def render(skill: dict[str, Any], arguments: str) -> str:
        path = Path(skill["path"])
        content = path.read_text(encoding="utf-8")
        match = re.match(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", content, re.DOTALL)
        if not match:
            raise ValueError(f"Skill file {path} is missing frontmatter")

        instructions = content[match.end():]
        if "$ARGUMENTS" in instructions:
            instructions = instructions.replace("$ARGUMENTS", arguments)
        elif arguments:
            instructions += f"\n\nArguments: {arguments}"

        return instructions

    @staticmethod
    def expand_skill_stack(
        prompt: str,
        skills: list[dict[str, Any]],
        execute: Callable[[dict[str, Any], str], str],
    ) -> str:
        available_skills = {
            name: skill for skill in skills
            for name in (skill["name"], Path(skill["path"]).parent.name)
        }

        remaining = prompt.lstrip()
        commands = []
        while remaining.startswith("/"):
            parts = remaining.split(maxsplit=1)
            token = parts[0]
            skill = available_skills.get(token[1:])

            if skill is None:
                break

            commands.append((token, skill))
            remaining = parts[1] if len(parts) > 1 else ""

        if not commands:
            return prompt

        instructions = [
            f'<skill name="{escape(skill["name"], quote=True)}">\n'
            + execute(Skills.find(command[1:], skills), remaining)
            + "\n</skill>"
            for command, skill in commands
        ]

        return (
            "Apply the following skill instructions and use any completed subagent results. "
            "Combine their findings into one response.\n\n"
            + "\n\n".join(instructions)
        )


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

    def clone(self) -> "ToolRegistry":
        registry = ToolRegistry()
        registry._tools = self._tools.copy()
        registry._functions = self._functions.copy()
        return registry

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
            depth: int = 0,
            max_depth: int = 3,
            ) -> None:
        self.client = client
        self.tools = tools.clone()
        self.model = model
        self.max_turns = max_turns
        self.depth = depth
        self.max_depth = max_depth

    def run(self, prompt: str) -> str:

        skills = Skills.load_skills_catalog()
        prompt = Skills.expand_skill_stack(
            prompt, skills, lambda skill, arguments: self.execute_skill(skill, arguments, skills)
        )
        return self.run_loop(prompt, skills)

    def execute_skill(
        self, skill: dict[str, Any], arguments: str, skills: list[dict[str, Any]]
    ) -> str:
        instructions = Skills.render(skill, arguments)
        base_directory = Path(skill["path"]).parent
        if skill.get("context") == "fork":
            if self.depth >= self.max_depth:
                raise RuntimeError(f"Maximum subagent depth ({self.max_depth}) reached")
            child = Agent(
                client=self.client,
                tools=self.tools,
                model=self.model,
                max_turns=self.max_turns,
                depth=self.depth + 1,
                max_depth=self.max_depth,
            )
            result = child.run_loop(
                instructions,
                skills,
                system_context=(
                    "You are a subagent. Complete the assigned skill task and return a final answer. "
                    f"Resolve relative skill file references from: {base_directory}"
                ),
            )
            return f"Completed subagent result for {skill['name']}:\n{result}"

        return (
            f"Follow the {skill['name']} skill instructions.\n"
            f"Resolve relative skill file references from: {base_directory}\n\n"
            f"{instructions}"
        )

    def run_loop(
        self, prompt: str, skills: list[dict[str, Any]], system_context: str = ""
    ) -> str:
        self.tools.register_tool(
            name="Skill",
            description=(
                "Load a skill's instructions with optional arguments. "
                "Choose a skill from the catalog. Regular skills return instructions; "
                "context: fork skills run a subagent and return its final result."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "description": "The skill name from the catalog"},
                    "args": {"type": "string", "description": "Arguments to substitute into the skill"},
                },
                "required": ["skill"],
            },
            function=lambda arguments: Skills.invoke(
                arguments, skills,
                lambda skill, args: self.execute_skill(skill, args, skills),
            ),
        )

        messages = [
            {"role": "system", "content": Skills.skills_prompt(skills) + "\n" + system_context},
            {"role": "user", "content": prompt}]
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
