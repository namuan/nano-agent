#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "openai",
# ]
# ///

"""
Agent that uses LLM to execute shell commands to find answers.
Stores data persistently between sessions.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import openai


PERSISTENT_STORAGE_PATH = Path.home() / ".agent_storage.json"
SYSTEM_PROMPT = """You are a helpful agent that uses shell commands to find information and answer questions.

You have access to the following tools:
- `shell <command>`: Execute a shell command and see its output
- `done <answer>`: Use this when you have found the answer and want to complete the task

IMPORTANT: Output exactly ONE THOUGHT and ONE ACTION per response. Do not output multiple actions at once.

Guidelines:
1. Think step by step about what information you need
2. Use shell commands to explore, search, and gather information
3. Be careful with destructive commands (rm, etc.) - avoid them unless necessary
4. Store useful information in memory for future reference
5. When you have the answer, use the `done` tool
6. For multiline commands (like heredocs), put everything on one line or use semicolons

Persistent storage is available for remembering data across sessions. Use it wisely.

Format your response EXACTLY as:
THOUGHT: Your reasoning about what to do next
ACTION: The action to take (shell <command> or done <answer>)

Example:
THOUGHT: I need to find files in the current directory
ACTION: shell ls -la

Example with multiline content:
THOUGHT: I will create a file with content
ACTION: shell echo "line1\nline2" > file.txt
"""


def load_storage() -> dict[str, Any]:
    """Load persistent storage."""
    if PERSISTENT_STORAGE_PATH.exists():
        with open(PERSISTENT_STORAGE_PATH) as f:
            return json.load(f)
    return {}


def save_storage(data: dict[str, Any]) -> None:
    """Save persistent storage."""
    PERSISTENT_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PERSISTENT_STORAGE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def execute_shell(command: str) -> str:
    """Execute a shell command and return the output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        stdout = result.stdout
        stderr = result.stderr
        if result.returncode != 0:
            return f"Exit code: {result.returncode}\nStdout: {stdout}\nStderr: {stderr}"
        return stdout or "Command executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds"
    except Exception as e:
        return f"Error executing command: {e}"


def parse_response(response: str) -> tuple[str, str]:
    """Parse the LLM response into thought and action."""
    thought = ""
    action = ""

    # Find the first THOUGHT: and first ACTION: in the response
    lines = response.split("\n")
    thought_lines = []
    action_lines = []
    in_thought = False
    in_action = False
    action_found = False

    for line in lines:
        line_stripped = line.strip()

        # Check for THOUGHT:
        if re.match(r"\*?\*?THOUGHT\*?\*?:", line_stripped, re.IGNORECASE):
            in_thought = True
            in_action = False
            thought_lines.append(
                re.sub(
                    r"\*?\*?THOUGHT\*?\*?:\s*", "", line_stripped, flags=re.IGNORECASE
                )
            )
        elif re.match(r"\*?\*?ACTION\*?\*?:", line_stripped, re.IGNORECASE):
            if action_found:
                # Stop at second ACTION - we only want the first one
                break
            in_thought = False
            in_action = True
            action_found = True
            action_lines.append(
                re.sub(
                    r"\*?\*?ACTION\*?\*?:\s*", "", line_stripped, flags=re.IGNORECASE
                )
            )
        elif in_thought:
            thought_lines.append(line)
        elif in_action:
            action_lines.append(line)

    thought = "\n".join(thought_lines).strip()
    action = "\n".join(action_lines).strip()

    return thought, action


def run_agent(prompt: str, base_url: str, model: str, max_iterations: int = 30) -> str:
    """Run the agent with the given prompt."""
    client = openai.OpenAI(
        base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "not-needed")
    )
    storage = load_storage()

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Prompt: {prompt}\n\nPersistent storage: {json.dumps(storage, indent=2)}",
        },
    ]

    for i in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=conversation,
            temperature=0.7,
        )

        content = response.choices[0].message.content
        print(f"\n[Iteration {i + 1}]\n{content}")

        thought, action = parse_response(content)
        print(
            f"\n[Parsed] Thought: {thought[:100]}{'...' if len(thought) > 100 else ''}"
        )
        print(f"[Parsed] Action: {action[:200]}{'...' if len(action) > 200 else ''}")

        if not action:
            print("\n[Error] No action found in response")
            conversation.append({"role": "assistant", "content": content})
            conversation.append(
                {
                    "role": "user",
                    "content": "I could not parse an action from your response. Please output exactly one THOUGHT and one ACTION per response. Format: THOUGHT: ... ACTION: shell ...",
                }
            )
        elif action.startswith("done "):
            answer = action.replace("done ", "", 1)
            save_storage(storage)
            return answer
        elif action.startswith("shell "):
            command = action.replace("shell ", "", 1)
            output = execute_shell(command)
            print(
                f"\n[Shell Output]\n{output[:500]}{'...' if len(output) > 500 else ''}"
            )

            conversation.append({"role": "assistant", "content": content})
            conversation.append(
                {
                    "role": "user",
                    "content": f"Output:\n{output}\n\nContinue with your next thought and action.",
                }
            )
        else:
            conversation.append({"role": "assistant", "content": content})
            conversation.append(
                {
                    "role": "user",
                    "content": "Invalid action format. Please use 'shell <command>' or 'done <answer>' format with THOUGHT: and ACTION: prefixes.",
                }
            )

    save_storage(storage)
    return "Maximum iterations reached without finding an answer."


def main():
    parser = argparse.ArgumentParser(
        description="Agent that uses LLM to execute shell commands"
    )
    parser.add_argument("prompt", help="The prompt/question to answer")
    parser.add_argument(
        "--endpoint",
        "-e",
        default="http://127.0.0.1:8080/v1",
        help="OpenAI-compatible endpoint URL",
    )
    parser.add_argument(
        "--model", "-m", default="qwen2.5:32b", help="Model name to use"
    )
    parser.add_argument(
        "--max-iterations",
        "-i",
        type=int,
        default=30,
        help="Maximum number of iterations",
    )

    args = parser.parse_args()

    answer = run_agent(args.prompt, args.endpoint, args.model, args.max_iterations)
    print(f"\n{'=' * 50}")
    print(f"FINAL ANSWER: {answer}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
