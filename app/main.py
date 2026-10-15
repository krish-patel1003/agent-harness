import argparse
import os
import sys
import json

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def reading_tool(args):
    with open(args['file_path'], 'r') as f:
        return f.read()


def get_tool(tool_name):
    tools_functions = {
            "Read": reading_tool
    }
    return tools_functions.get(tool_name, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages = [{"role": "user", "content": args.p}]

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)

    tools = {
        "Read": {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }
    }


    ReAct = True

    while ReAct:
        chat = client.chat.completions.create(
            model = "anthropic/claude-haiku-4.5", messages = messages, tools = tools.values())

        message = chat.choices[0].message
        messages.append(message)

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        if chat.choices[0].message.tool_calls:

            for call in chat.choices[0].message.tool_calls:

                tool_key = call.function.name
                arguments = json.loads(call.function.arguments)
                function = get_tool(tool_key)
                result = function(arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                })

        else:
            print(chat.choices[0].message.content)
            break


if __name__ == "__main__":
    main()
