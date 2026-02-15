#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "pyyaml",
#   "requests",
#   "jinja2",
#   "pydantic >= 2.0",
#   "litellm >= 1.75.5",
#   "tenacity",
#   "rich",
#   "python-dotenv",
#   "typer",
#   "platformdirs",
#   "textual",
#   "prompt_toolkit",
#   "datasets",
#   "openai != 1.100.0,!=1.100.1"
# ]
# ///

"""
Complete mini-swe-agent implementation in a single file.
Generated from the mini-swe-agent repository: https://github.com/SWE-agent/mini-SWE-agent

This file contains all the actual implementation code from each Python file in src/minisweagent/
with proper module structure and imports.
"""

# ==============================
# Core Module - minisweagent
# ==============================

__version__ = "2.1.0"

import os
import time
from pathlib import Path
from typing import Any, Protocol

import dotenv
from platformdirs import user_config_dir
from rich.console import Console


package_dir = Path(__file__).resolve().parent


# === Protocols ===


class Model(Protocol):
    """Protocol for language models."""

    config: Any

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict: ...

    def format_message(self, **kwargs) -> dict: ...

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...


class Environment(Protocol):
    """Protocol for execution environments."""

    config: Any

    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...


class Agent(Protocol):
    """Protocol for agents."""

    config: Any

    def run(self, task: str, **kwargs) -> dict: ...

    def save(self, path: Path | None, *extra_dicts) -> dict: ...


# ==============================
# Exceptions
# ==============================


class InterruptAgentFlow(Exception):
    """Raised to interrupt the agent flow and add messages."""

    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


class Submitted(InterruptAgentFlow):
    """Raised when the agent has completed its task."""


class LimitsExceeded(InterruptAgentFlow):
    """Raised when the agent has exceeded its cost or step limit."""


class UserInterruption(InterruptAgentFlow):
    """Raised when the user interrupts the agent."""


class FormatError(InterruptAgentFlow):
    """Raised when the LM's output is not in the expected format."""


# ==============================
# Utils - log
# ==============================

import logging
from pathlib import Path

from rich.logging import RichHandler


def _setup_root_logger() -> None:
    logger = logging.getLogger("minisweagent")
    logger.setLevel(logging.DEBUG)
    _handler = RichHandler(
        show_path=False,
        show_time=False,
        show_level=False,
        markup=True,
    )
    _formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)


def add_file_handler(
    path: Path | str, level: int = logging.DEBUG, *, print_path: bool = True
) -> None:
    logger = logging.getLogger("minisweagent")
    handler = logging.FileHandler(path)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if print_path:
        print(f"Logging to '{path}'")


_setup_root_logger()
logger = logging.getLogger("minisweagent")


# ==============================
# Utils - serialize
# ==============================

UNSET = object()


def recursive_merge(*dictionaries: dict | None) -> dict:
    """Merge multiple dictionaries recursively.

    Later dictionaries take precedence over earlier ones.
    Nested dictionaries are merged recursively.
    UNSET values are skipped.
    """
    if not dictionaries:
        return {}
    result: dict[str, Any] = {}
    for d in dictionaries:
        if d is None:
            continue
        for key, value in d.items():
            if value is UNSET:
                continue
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = recursive_merge(result[key], value)
            elif isinstance(value, dict):
                result[key] = recursive_merge(value)
            else:
                result[key] = value
    return result


# ==============================
# Agents
# ==============================

import copy
import importlib

_AGENT_MAPPING = {
    "default": "DefaultAgent",
    "interactive": "InteractiveAgent",
}


def get_agent_class(spec: str) -> type[Agent]:
    class_name = _AGENT_MAPPING.get(spec, spec)
    # Try to find the class in the current module
    try:
        import sys

        current_module = sys.modules[__name__]
        return getattr(current_module, class_name)
    except AttributeError:
        msg = f"Unknown agent type: {spec} (available: {_AGENT_MAPPING})"
        raise ValueError(msg)


def get_agent(
    model: Model, env: Environment, config: dict, *, default_type: str = ""
) -> Agent:
    config = copy.deepcopy(config)
    agent_class = get_agent_class(config.pop("agent_class", default_type))
    return agent_class(model, env, **config)


# ==============================
# Agents - default
# ==============================

import json
import logging
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    output_path: Path | None = None
    """Save the trajectory to this path."""


class DefaultAgent:
    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: type = AgentConfig,
        **kwargs,
    ):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {"n_model_calls": self.n_calls, "model_cost": self.cost},
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(
            **self.get_template_vars()
        )

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(
                role="system",
                content=self._render_template(self.config.system_template),
            ),
            self.model.format_message(
                role="user",
                content=self._render_template(self.config.instance_template),
            ),
        )
        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        return self.execute_actions(self.query())

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if (
            0 < self.config.step_limit <= self.n_calls
            or 0 < self.config.cost_limit <= self.cost
        ):
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        outputs = [
            self.env.execute(action)
            for action in message.get("extra", {}).get("actions", [])
        ]
        return self.add_messages(
            *self.model.format_observation_messages(
                message, outputs, self.get_template_vars()
            )
        )

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(
            agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts
        )

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data


# ==============================
# Agents - interactive
# ==============================

import re
from typing import Literal, NoReturn

from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console
from rich.rule import Rule


console = Console(highlight=False)
_history = FileHistory(Path(__file__).resolve().parent / "interactive_history.txt")
_prompt_session = PromptSession(history=_history)
_multiline_prompt_session = PromptSession(history=_history, multiline=True)


class InteractiveAgentConfig(AgentConfig):
    mode: Literal["human", "confirm", "yolo"] = "confirm"
    """Whether to confirm actions."""
    whitelist_actions: list[str] = []
    """Never confirm actions that match these regular expressions."""
    confirm_exit: bool = True
    """If the agent wants to finish, do we ask for confirmation from user?"""


def _multiline_prompt() -> str:
    return _multiline_prompt_session.prompt(
        "",
        bottom_toolbar=HTML(
            "Submit message: <b fg='yellow' bg='black'>Esc, then Enter</b> | "
            "Navigate history: <b fg='yellow' bg='black'>Arrow Up/Down</b> | "
            "Search history: <b fg='yellow' bg='black'>Ctrl+R</b>"
        ),
    )


class InteractiveAgent(DefaultAgent):
    _MODE_COMMANDS_MAPPING = {"/u": "human", "/c": "confirm", "/y": "yolo"}

    def __init__(self, *args, config_class=InteractiveAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.cost_last_confirmed = 0.0

    def _interrupt(self, content: str, *, itype: str = "UserInterruption") -> NoReturn:
        raise UserInterruption(
            {"role": "user", "content": content, "extra": {"interrupt_type": itype}}
        )

    def add_messages(self, *messages: dict) -> list[dict]:
        for msg in messages:
            role, content = (
                msg.get("role") or msg.get("type", "unknown"),
                get_content_string(msg),
            )
            if role == "assistant":
                console.print(
                    f"\n[red][bold]mini-swe-agent[/bold] (step [bold]{self.n_calls}[/bold], [bold]${self.cost:.2f}[/bold]):[/red]\n",
                    end="",
                    highlight=False,
                )
            else:
                console.print(
                    f"\n[bold green]{role.capitalize()}[/bold green]:\n",
                    end="",
                    highlight=False,
                )
            console.print(content, highlight=False, markup=False)
        return super().add_messages(*messages)

    def query(self) -> dict:
        if self.config.mode == "human":
            match command := self._prompt_and_handle_slash_commands(
                "[bold yellow]>[/bold yellow] "
            ):
                case "/y" | "/c":
                    pass
                case _:
                    msg = {
                        "role": "user",
                        "content": f"User command: \n```bash\n{command}\n```",
                        "extra": {"actions": [{"command": command}]},
                    }
                    self.add_messages(msg)
                    return msg
        try:
            with console.status("Waiting for the LM to respond..."):
                return super().query()
        except LimitsExceeded:
            console.print(
                f"Limits exceeded. Limits: {self.config.step_limit} steps, ${self.config.cost_limit}.\n"
                f"Current spend: {self.n_calls} steps, ${self.cost:.2f}."
            )
            self.config.step_limit = int(input("New step limit: "))
            self.config.cost_limit = float(input("New cost limit: "))
            return super().query()

    def step(self) -> list[dict]:
        try:
            console.print(Rule())
            return super().step()
        except KeyboardInterrupt:
            interruption_message = self._prompt_and_handle_slash_commands(
                "\n\n[bold yellow]Interrupted.[/bold yellow] "
                "[green]Type a comment/command[/green] (/h for available commands)"
                "\n[bold yellow]>[/bold yellow] "
            ).strip()
            if (
                not interruption_message
                or interruption_message in self._MODE_COMMANDS_MAPPING
            ):
                interruption_message = "Temporary interruption caught."
            self._interrupt(f"Interrupted by user: {interruption_message}")

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        commands = [action["command"] for action in actions]
        outputs = []
        try:
            self._ask_confirmation_or_interrupt(commands)
            for action in actions:
                outputs.append(self.env.execute(action))
        except Submitted as e:
            self._check_for_new_task_or_submit(e)
        finally:
            result = self.add_messages(
                *self.model.format_observation_messages(
                    message, outputs, self.get_template_vars()
                )
            )
        return result

    def _add_observation_messages(
        self, message: dict, outputs: list[dict]
    ) -> list[dict]:
        return self.add_messages(
            *self.model.format_observation_messages(
                message, outputs, self.get_template_vars()
            )
        )

    def _check_for_new_task_or_submit(self, e: Submitted) -> NoReturn:
        if self.config.confirm_exit:
            message = (
                "[bold yellow]Agent wants to finish.[/bold yellow] "
                "[bold green]Type new task[/bold green] or [bold]Enter[/bold] to quit "
                "([bold]/h[/bold] for commands)\n"
                "[bold yellow]>[/bold yellow] "
            )
            user_input = self._prompt_and_handle_slash_commands(message).strip()
            if user_input == "/u":
                self._interrupt("Switched to human mode.")
            elif user_input in self._MODE_COMMANDS_MAPPING:
                return self._check_for_new_task_or_submit(e)
            elif user_input:
                self._interrupt(
                    f"The user added a new task: {user_input}", itype="UserNewTask"
                )
        raise e

    def _should_ask_confirmation(self, action: str) -> bool:
        return self.config.mode == "confirm" and not any(
            re.match(r, action) for r in self.config.whitelist_actions
        )

    def _ask_confirmation_or_interrupt(self, commands: list[str]) -> None:
        if not any(self._should_ask_confirmation(c) for c in commands):
            return
        prompt = (
            f"[bold yellow]Execute {len(commands)} action(s)?[/] [green][bold]Enter[/] to confirm[/], "
            "[red]type [bold]comment[/] to reject[/], or [blue][bold]/h[/] to show available commands[/]\n"
            "[bold yellow]>[/bold yellow] "
        )
        match user_input := self._prompt_and_handle_slash_commands(prompt).strip():
            case "" | "/y":
                pass
            case "/u":
                self._interrupt(
                    "Commands not executed. Switching to human mode",
                    itype="UserRejection",
                )
            case _:
                self._interrupt(
                    f"Commands not executed. The user rejected your commands with the following message: {user_input}",
                    itype="UserRejection",
                )

    def _prompt_and_handle_slash_commands(
        self, prompt: str, *, _multiline: bool = False
    ) -> str:
        console.print(prompt, end="")
        if _multiline:
            return _multiline_prompt()
        user_input = _prompt_session.prompt("")
        if user_input == "/m":
            return self._prompt_and_handle_slash_commands(prompt, _multiline=True)
        if user_input == "/h":
            console.print(
                f"Current mode: [bold green]{self.config.mode}[/bold green]\n"
                f"[bold green]/y[/bold green] to switch to [bold yellow]yolo[/bold yellow] mode (execute LM commands without confirmation)\n"
                f"[bold green]/c[/bold green] to switch to [bold yellow]confirmation[/bold yellow] mode (ask for confirmation before executing LM commands)\n"
                f"[bold green]/u[/bold green] to switch to [bold yellow]human[/bold yellow] mode (execute commands issued by the user)\n"
                f"[bold green]/m[/bold green] to enter multiline comment",
            )
            return self._prompt_and_handle_slash_commands(prompt)
        if user_input in self._MODE_COMMANDS_MAPPING:
            if self.config.mode == self._MODE_COMMANDS_MAPPING[user_input]:
                return self._prompt_and_handle_slash_commands(
                    f"[bold red]Already in {self.config.mode} mode.[/bold red]\n{prompt}"
                )
            self.config.mode = self._MODE_COMMANDS_MAPPING[user_input]
            console.print(
                f"Switched to [bold green]{self.config.mode}[/bold green] mode."
            )
            return user_input
        return user_input


# ==============================
# Environments
# ==============================

_ENVIRONMENT_MAPPING = {
    "docker": "DockerEnvironment",
    "singularity": "SingularityEnvironment",
    "local": "LocalEnvironment",
    "swerex_docker": "SwerexDockerEnvironment",
    "swerex_modal": "SwerexModalEnvironment",
    "bubblewrap": "BubblewrapEnvironment",
}


def get_environment_class(spec: str) -> type[Environment]:
    class_name = _ENVIRONMENT_MAPPING.get(spec, spec)
    # Try to find the class in the current module
    try:
        import sys

        current_module = sys.modules[__name__]
        return getattr(current_module, class_name)
    except AttributeError:
        msg = f"Unknown environment type: {spec} (available: {_ENVIRONMENT_MAPPING})"
        raise ValueError(msg)


def get_environment(config: dict, *, default_type: str = "") -> Environment:
    config = copy.deepcopy(config)
    environment_class = config.pop("environment_class", default_type)
    return get_environment_class(environment_class)(**config)


# ==============================
# Environments - local
# ==============================

import platform
import subprocess


class LocalEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30


class LocalEnvironment:
    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        """This class executes bash commands directly on the local machine."""
        self.config = config_class(**kwargs)

    def execute(
        self, action: dict, cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        """Execute a command in the local environment and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd or os.getcwd()
        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                cwd=cwd,
                env=os.environ | self.config.env,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace")
                if isinstance(raw_output, bytes)
                else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(
            self.config.model_dump(), platform.uname()._asdict(), os.environ, kwargs
        )

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


# ==============================
# Environments - docker
# ==============================

import shlex
import uuid


class DockerEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    env: dict[str, str] = {}
    forward_env: list[str] = []
    timeout: int = 30
    executable: str = "docker"
    run_args: list[str] = ["--rm"]
    container_timeout: str = "2h"
    pull_timeout: int = 120
    interpreter: list[str] = ["bash", "-lc"]


class DockerEnvironment:
    def __init__(
        self,
        *,
        config_class: type = DockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        self._start_container()

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(
            self.config.model_dump(), platform.uname()._asdict(), kwargs
        )

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        container_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
        cmd = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ]
        self.logger.debug(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,
            check=True,
        )
        self.logger.info(
            f"Started container {container_name} with ID {result.stdout.strip()}"
        )
        self.container_id = result.stdout.strip()

    def execute(
        self, action: dict, cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        cmd = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, *self.config.interpreter, command])

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace")
                if isinstance(raw_output, bytes)
                else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        if getattr(self, "container_id", None) is not None:
            cmd = f"(timeout 60 {self.config.executable} stop {self.container_id} || {self.config.executable} rm -f {self.container_id}) >/dev/null 2>&1 &"
            subprocess.Popen(cmd, shell=True)

    def __del__(self):
        self.cleanup()


# ==============================
# Environments - singularity
# ==============================

import shutil
import tempfile


class SingularityEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    env: dict[str, str] = {}
    forward_env: list[str] = []
    timeout: int = 30
    executable: str = "singularity"
    sandbox_build_retries: int = 3
    global_args: list[str] = ["--quiet"]
    exec_args: list[str] = ["--contain", "--cleanenv", "--fakeroot"]


class SingularityEnvironment:
    def __init__(
        self,
        *,
        config_class: type = SingularityEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.sandbox_dir = self._build_sandbox()

    def _build_sandbox(self) -> Path:
        max_retries = self.config.sandbox_build_retries
        for attempt in range(max_retries):
            sandbox_dir = (
                Path(tempfile.gettempdir()) / f"minisweagent-{uuid.uuid4().hex[:8]}"
            )
            try:
                subprocess.run(
                    [
                        self.config.executable,
                        "build",
                        "--sandbox",
                        sandbox_dir,
                        self.config.image,
                    ],
                    check=True,
                    capture_output=True,
                )
                break
            except subprocess.CalledProcessError as e:
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                self.logger.error(
                    f"Error building image {self.config.image}, stdout: {e.stdout}, stderr: {e.stderr} (attempt {attempt + 1}/{max_retries})"
                )
                if attempt == max_retries - 1:
                    raise
        return sandbox_dir

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def execute(
        self, action: dict, cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        command = action.get("command", "")
        cmd = [
            self.config.executable,
            *self.config.global_args,
            "exec",
            *self.config.exec_args,
        ]

        work_dir = cwd or self.config.cwd
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])

        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.extend(["--writable", str(self.sandbox_dir), "bash", "-c", command])
        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace")
                if isinstance(raw_output, bytes)
                else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def __del__(self):
        self.cleanup()


# ==============================
# Environments - bubblewrap
# ==============================


class BubblewrapEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30
    executable: str = "bwrap"
    wrapper_args: list[str] = [
        "--unshare-user-try",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/bin",
        "/bin",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/etc",
        "/etc",
        "--tmpfs",
        "/tmp",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--new-session",
        "--setenv",
        "PATH",
        "/usr/local/bin:/usr/sbin:/usr/bin:/bin",
    ]


class BubblewrapEnvironment:
    def __init__(
        self,
        *,
        config_class: type = BubblewrapEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.working_dir = (
            Path(tempfile.gettempdir()) / f"minisweagent-{uuid.uuid4().hex[:8]}"
        )
        self.working_dir.mkdir(parents=True)

    def execute(
        self, action: dict, cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        command = action.get("command", "")
        cwd = cwd or self.config.cwd or str(self.working_dir)

        cmd = (
            [self.config.executable]
            + self.config.wrapper_args
            + ["--bind", cwd, cwd, "--chdir", cwd]
        )

        for key, value in self.config.env.items():
            cmd.extend(["--setenv", key, value])

        cmd.extend(["bash", "-c", command])

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace")
                if isinstance(raw_output, bytes)
                else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)

    def __del__(self):
        self.cleanup()

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(
            self.config.model_dump(), platform.uname()._asdict(), kwargs
        )

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


# ==============================
# Environments - swerex docker
# ==============================

import asyncio


class SwerexDockerEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    timeout: int = 30
    deployment_extra_kwargs: dict[str, Any] = {}


class SwerexDockerEnvironment:
    def __init__(self, **kwargs):
        self.config = SwerexDockerEnvironmentConfig(**kwargs)
        try:
            from swerex.deployment.docker import DockerDeployment
            from swerex.runtime.abstract import Command as RexCommand

            self.deployment = DockerDeployment(
                image=self.config.image, **self.config.deployment_extra_kwargs
            )
            asyncio.run(self.deployment.start())
        except ImportError:
            raise ImportError(
                "The swerex package is required to use SwerexDockerEnvironment. Please install it with: pip install swe-rex"
            )

    def execute(
        self, action: dict, cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        command = action.get("command", "")
        try:
            from swerex.runtime.abstract import Command as RexCommand

            result = asyncio.run(
                self.deployment.runtime.execute(
                    RexCommand(
                        command=command,
                        shell=True,
                        check=False,
                        cwd=cwd or self.config.cwd,
                        timeout=timeout or self.config.timeout,
                        merge_output_streams=True,
                    )
                )
            )
            output = {
                "output": result.stdout,
                "returncode": result.exit_code,
                "exception_info": "",
            }
        except Exception as e:
            output = {
                "output": str(e) if str(e) else "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


# ==============================
# Environments - swerex modal
# ==============================


class SwerexModalEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    timeout: int = 30
    env: dict[str, str] = {}
    startup_timeout: float = 60.0
    runtime_timeout: float = 3600.0
    deployment_timeout: float = 3600.0
    install_pipx: bool = True
    modal_sandbox_kwargs: dict[str, Any] = {}


class SwerexModalEnvironment:
    def __init__(self, **kwargs):
        self.config = SwerexModalEnvironmentConfig(**kwargs)
        try:
            from swerex.deployment.modal import ModalDeployment
            from swerex.runtime.abstract import Command as RexCommand

            self.deployment = ModalDeployment(
                image=self.config.image,
                startup_timeout=self.config.startup_timeout,
                runtime_timeout=self.config.runtime_timeout,
                deployment_timeout=self.config.deployment_timeout,
                install_pipx=self.config.install_pipx,
                modal_sandbox_kwargs=self.config.modal_sandbox_kwargs,
            )
            asyncio.run(self.deployment.start())
        except ImportError:
            raise ImportError(
                "The swerex package is required to use SwerexModalEnvironment. Please install it with: pip install swe-rex"
            )

    def execute(
        self, command: str, cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        from swerex.runtime.abstract import Command as RexCommand

        output = asyncio.run(
            self.deployment.runtime.execute(
                RexCommand(
                    command=command,
                    shell=True,
                    check=False,
                    cwd=cwd or self.config.cwd,
                    timeout=timeout or self.config.timeout,
                    merge_output_streams=True,
                    env=self.config.env if self.config.env else None,
                )
            )
        )
        return {
            "output": output.stdout,
            "returncode": output.exit_code,
        }

    def get_template_vars(self) -> dict[str, Any]:
        return self.config.model_dump()

    def stop(self):
        async def _stop():
            await asyncio.wait_for(self.deployment.stop(), timeout=10)

        try:
            asyncio.run(_stop())
        except Exception:
            pass


# ==============================
# Models - global stats
# ==============================

import threading


class GlobalModelStats:
    """Global model statistics tracker with optional limits."""

    def __init__(self):
        self._cost = 0.0
        self._n_calls = 0
        self._lock = threading.Lock()
        self.cost_limit = 0.0
        self.call_limit = 0

    def add(self, cost: float) -> None:
        """Add a model call with its cost, checking limits."""
        with self._lock:
            self._cost += cost
            self._n_calls += 1
        if 0 < self.cost_limit < self._cost or 0 < self.call_limit < self._n_calls + 1:
            raise RuntimeError(
                f"Global cost/call limit exceeded: ${self._cost:.4f} / {self._n_calls}"
            )

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def n_calls(self) -> int:
        return self._n_calls


GLOBAL_MODEL_STATS = GlobalModelStats()


def get_model(input_model_name: str | None = None, config: dict | None = None) -> Model:
    """Get an initialized model object from any kind of user input or settings."""
    resolved_model_name = get_model_name(input_model_name, config)
    if config is None:
        config = {}
    config = copy.deepcopy(config)
    config["model_name"] = resolved_model_name

    model_class = get_model_class(resolved_model_name, config.pop("model_class", ""))

    if (
        any(
            s in resolved_model_name.lower()
            for s in ["anthropic", "sonnet", "opus", "claude"]
        )
        and "set_cache_control" not in config
    ):
        config["set_cache_control"] = "default_end"

    return model_class(**config)


def get_model_name(
    input_model_name: str | None = None, config: dict | None = None
) -> str:
    if config is None:
        config = {}
    if input_model_name:
        return input_model_name
    if from_config := config.get("model_name"):
        return from_config
    raise ValueError("No default model set. Please specify a model name.")


_MODEL_CLASS_MAPPING = {
    "litellm": "LitellmModel",
    "litellm_textbased": "LitellmTextbasedModel",
    "litellm_response": "LitellmResponseModel",
    "openrouter": "OpenRouterModel",
    "openrouter_textbased": "OpenRouterTextbasedModel",
    "openrouter_response": "OpenRouterResponseModel",
    "portkey": "PortkeyModel",
    "portkey_response": "PortkeyResponseAPIModel",
    "requesty": "RequestyModel",
    "deterministic": "DeterministicModel",
}


def get_model_class(model_name: str, model_class: str = "") -> type:
    if model_class:
        class_name = _MODEL_CLASS_MAPPING.get(model_class, model_class)
        # Try to find the class in the current module
        try:
            import sys

            current_module = sys.modules[__name__]
            return getattr(current_module, class_name)
        except AttributeError:
            msg = f"Unknown model class: {model_class} (available: {_MODEL_CLASS_MAPPING})"
            raise ValueError(msg)

    # Since all code is in a single file, we can return the LitellmModel class directly
    return LitellmModel


# ==============================
# Models - utils - anthropic utils
# ==============================


def _is_anthropic_thinking_block(block) -> bool:
    if not isinstance(block, dict):
        return False
    return block.get("type") in ("thinking", "redacted_thinking")


def _reorder_anthropic_thinking_blocks(messages: list[dict]) -> list[dict]:
    result = []
    for msg in messages:
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            content = msg["content"]
            thinking_blocks = [b for b in content if _is_anthropic_thinking_block(b)]
            if thinking_blocks:
                other_blocks = [
                    b for b in content if not _is_anthropic_thinking_block(b)
                ]
                if other_blocks:
                    msg = {**msg, "content": thinking_blocks + other_blocks}
                else:
                    msg = {
                        **msg,
                        "content": thinking_blocks + [{"type": "text", "text": ""}],
                    }
        result.append(msg)
    return result


# ==============================
# Models - utils - cache control
# ==============================

import copy
import warnings
from typing import Literal


def _get_content_text(entry: dict) -> str | None:
    if entry["content"] is None:
        return None
    if isinstance(entry["content"], str):
        return entry["content"]
    assert len(entry["content"]) == 1, "Expected single message in content"
    return entry["content"][0]["text"]


def _clear_cache_control(entry: dict) -> None:
    if isinstance(entry["content"], list):
        assert len(entry["content"]) == 1, "Expected single message in content"
        entry["content"][0].pop("cache_control", None)
    entry.pop("cache_control", None)


def _set_cache_control(entry: dict) -> None:
    if entry["content"] is None:
        entry["cache_control"] = {"type": "ephemeral"}
        return

    if not isinstance(entry["content"], list):
        entry["content"] = [
            {
                "type": "text",
                "text": _get_content_text(entry),
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        entry["content"][0]["cache_control"] = {"type": "ephemeral"}
    if entry["role"] == "tool":
        entry["content"][0].pop("cache_control", None)
        entry["cache_control"] = {"type": "ephemeral"}


def set_cache_control(
    messages: list[dict],
    *,
    mode: Literal["default_end"] | None = "default_end",
    last_n_messages_offset: int = 0,
) -> list[dict]:
    if mode is None:
        return messages
    if mode != "default_end":
        raise ValueError(f"Invalid mode: {mode}")
    if last_n_messages_offset:
        warnings.warn(
            "last_n_messages_offset is deprecated and will be removed in the future. It has no effect."
        )

    messages = copy.deepcopy(messages)
    new_messages = []
    for i_entry, entry in enumerate(reversed(messages)):
        _clear_cache_control(entry)
        if i_entry == 0:
            _set_cache_control(entry)
        new_messages.append(entry)
    return list(reversed(new_messages))


# ==============================
# Models - utils - content string
# ==============================


def _format_tool_call(args_str: str) -> str:
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
        if isinstance(args, dict) and "command" in args:
            return f"```\n{args['command']}\n```"
    except Exception:
        pass
    return f"```\n{args_str}\n```"


def _format_observation(content: str) -> str | None:
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "returncode" in data:
            lines = []
            for key, value in data.items():
                lines.append(f"<{key}>")
                lines.append(str(value))
            return "\n".join(lines)
        return content
    except Exception:
        return content


def get_content_string(message: dict) -> str:
    texts = []

    content = message.get("content")
    if isinstance(content, str):
        texts.append(_format_observation(content))
    elif isinstance(content, list):
        texts.append(
            "\n".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        )

    if tool_calls := message.get("tool_calls"):
        for tc in tool_calls:
            func = (
                tc.get("function", {})
                if isinstance(tc, dict)
                else getattr(tc, "function", None)
            )
            if func:
                args = (
                    func.get("arguments", "{}")
                    if isinstance(func, dict)
                    else getattr(func, "arguments", "{}")
                )
                texts.append(_format_tool_call(args))

    if output := message.get("output"):
        if isinstance(output, str):
            texts.append(_format_observation(output))
        elif isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if isinstance(c, dict) and (text := c.get("text")):
                            texts.append(text)
                elif item.get("type") == "function_call":
                    texts.append(_format_tool_call(item.get("arguments", "{}")))

    return "\n\n".join(t for t in texts if t)


# ==============================
# Models - utils - openai multimodal
# ==============================

DEFAULT_MULTIMODAL_REGEX = r"(?s)<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>(.+?)</CONTENT_TYPE>(.+?)</MSWEA_MULTIMODAL_CONTENT>"


def _expand_content_string(*, content: str, pattern: str) -> list[dict]:
    matches = list(re.finditer(pattern, content))
    if not matches:
        return [{"type": "text", "text": content}]
    result = []
    last_end = 0
    for match in matches:
        text_before = content[last_end : match.start()]
        if text_before:
            result.append({"type": "text", "text": text_before})
        content_type = match.group(1).strip()
        extracted = match.group(2).strip()
        if content_type == "image_url":
            result.append({"type": "image_url", "image_url": {"url": extracted}})
        last_end = match.end()
    text_after = content[last_end:]
    if text_after:
        result.append({"type": "text", "text": text_after})
    return result


def expand_multimodal_content(content: Any, *, pattern: str) -> Any:
    if not pattern:
        return content
    content = copy.deepcopy(content)
    if isinstance(content, str):
        return _expand_content_string(content=content, pattern=pattern)
    if isinstance(content, list):
        return [expand_multimodal_content(item, pattern=pattern) for item in content]
    if isinstance(content, dict):
        if "content" not in content:
            return content
        content["content"] = expand_multimodal_content(
            content["content"], pattern=pattern
        )
        return content
    return str(content)


# ==============================
# Models - utils - retry
# ==============================

from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def retry(
    *, logger: logging.Logger, abort_exceptions: list[type[Exception]]
) -> Retrying:
    return Retrying(
        reraise=True,
        stop=stop_after_attempt(
            int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))
        ),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(tuple(abort_exceptions)),
    )


# ==============================
# Models - utils - actions text
# ==============================


def parse_regex_actions(
    content: str, *, action_regex: str, format_error_template: str
) -> list[dict]:
    actions = [a.strip() for a in re.findall(action_regex, content, re.DOTALL)]
    if len(actions) != 1:
        error_msg = f"Expected exactly 1 action, found {len(actions)}."
        raise FormatError(
            {
                "role": "user",
                "content": Template(
                    format_error_template, undefined=StrictUndefined
                ).render(actions=actions, error=error_msg),
                "extra": {
                    "interrupt_type": "FormatError",
                    "n_actions": len(actions),
                    "model_response": content,
                },
            }
        )
    return [{"command": action} for action in actions]


def format_observation_messages(
    outputs: list[dict],
    *,
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    results = []
    for output in outputs:
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg: dict = {
            "role": "user",
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results


# ==============================
# Models - utils - actions toolcall
# ==============================

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


def parse_toolcall_actions(
    tool_calls: list, *, format_error_template: str
) -> list[dict]:
    if not tool_calls:
        raise FormatError(
            {
                "role": "user",
                "content": Template(
                    format_error_template, undefined=StrictUndefined
                ).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call."
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    actions = []
    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}. "
        if tool_call.function.name != "bash":
            error_msg += f"Unknown tool '{tool_call.function.name}'."
        if "command" not in args:
            error_msg += "Missing 'command' argument in bash tool call."
        if error_msg:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(
                        format_error_template, undefined=StrictUndefined
                    ).render(error=error_msg.strip()),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        actions.append({"command": args["command"], "tool_call_id": tool_call.id})
    return actions


def format_toolcall_observation_messages(
    *,
    actions: list[dict],
    outputs: list[dict],
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    not_executed = {
        "output": "",
        "returncode": -1,
        "exception_info": "action was not executed",
    }
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs):
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg = {
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if "tool_call_id" in action:
            msg["tool_call_id"] = action["tool_call_id"]
            msg["role"] = "tool"
        else:
            msg["role"] = "user"
        if multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results


# ==============================
# Models - utils - actions toolcall response
# ==============================

BASH_TOOL_RESPONSE_API = {
    "type": "function",
    "name": "bash",
    "description": "Execute a bash command",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute",
            }
        },
        "required": ["command"],
    },
}


def _format_error_message(error_text: str) -> dict:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": error_text}],
        "extra": {"interrupt_type": "FormatError"},
    }


def parse_toolcall_actions_response(
    output: list, *, format_error_template: str
) -> list[dict]:
    tool_calls = []
    for item in output:
        item_type = (
            item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        )
        if item_type == "function_call":
            tool_calls.append(
                item.model_dump()
                if hasattr(item, "model_dump")
                else dict(item)
                if not isinstance(item, dict)
                else item
            )
    if not tool_calls:
        error_text = Template(format_error_template, undefined=StrictUndefined).render(
            error="No tool calls found in the response. Every response MUST include at least one tool call.",
        )
        raise FormatError(_format_error_message(error_text))
    actions = []
    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        try:
            args = json.loads(tool_call.get("arguments", "{}"))
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}. "
        if tool_call.get("name") != "bash":
            error_msg += f"Unknown tool '{tool_call.get('name')}'."
        if "command" not in args:
            error_msg += "Missing 'command' argument in bash tool call."
        if error_msg:
            error_text = Template(
                format_error_template, undefined=StrictUndefined
            ).render(error=error_msg.strip())
            raise FormatError(_format_error_message(error_text))
        actions.append(
            {
                "command": args["command"],
                "tool_call_id": tool_call.get("call_id") or tool_call.get("id"),
            }
        )
    return actions


def format_toolcall_observation_messages_response(
    *,
    actions: list[dict],
    outputs: list[dict],
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    not_executed = {
        "output": "",
        "returncode": -1,
        "exception_info": "action was not executed",
    }
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs):
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg: dict = {
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if "tool_call_id" in action:
            msg["type"] = "function_call_output"
            msg["call_id"] = action["tool_call_id"]
            msg["output"] = content
        else:
            msg["type"] = "message"
            msg["role"] = "user"
            msg["content"] = [{"type": "input_text", "text": content}]
        results.append(msg)
    return results


# ==============================
# Models - litellm model
# ==============================

import litellm


class LitellmModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    set_cache_control: Literal["default_end"] | None = None
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv(
        "MSWEA_COST_TRACKING", "default"
    )
    format_error_template: str = "{{ error }}"
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""


class LitellmModel:
    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Any = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        if (
            self.config.litellm_model_registry
            and Path(self.config.litellm_model_registry).is_file()
        ):
            litellm.utils.register_model(
                json.loads(Path(self.config.litellm_model_registry).read_text())
            )

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            # Set default API base to local OpenAI-compatible endpoint
            model_kwargs = self.config.model_kwargs | kwargs
            if "api_base" not in model_kwargs:
                model_kwargs["api_base"] = "http://127.0.0.1:8080/v1"
            # Ensure we have an API key (any value works)
            if "api_key" not in model_kwargs:
                model_kwargs["api_key"] = "placeholder-key"

            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL],
                **model_kwargs,
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(
            logger=logging.getLogger("litellm_model"),
            abort_exceptions=self.abort_exceptions,
        ):
            with attempt:
                response = self._query(
                    self._prepare_messages_for_api(messages), **kwargs
                )
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _calculate_cost(self, response) -> dict[str, float]:
        try:
            cost = litellm.cost_calculator.completion_cost(
                response, model=self.config.model_name
            )
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logging.getLogger("litellm_model").critical(msg)
                raise RuntimeError(msg) from e
        return {"cost": cost}

    def _parse_actions(self, response) -> list[dict]:
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(
            tool_calls, format_error_template=self.config.format_error_template
        )

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


# ==============================
# Models - litellm textbased model
# ==============================


class LitellmTextbasedModelConfig(LitellmModelConfig):
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions."


class LitellmTextbasedModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=LitellmTextbasedModelConfig, **kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            # Set default API base to local OpenAI-compatible endpoint
            model_kwargs = self.config.model_kwargs | kwargs
            if "api_base" not in model_kwargs:
                model_kwargs["api_base"] = "http://127.0.0.1:8080/v1"
            # Ensure we have an API key (any value works)
            if "api_key" not in model_kwargs:
                model_kwargs["api_key"] = "placeholder-key"

            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                **model_kwargs,
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _parse_actions(self, response: dict) -> list[dict]:
        content = response.choices[0].message.content or ""
        return parse_regex_actions(
            content,
            action_regex=self.config.action_regex,
            format_error_template=self.config.format_error_template,
        )

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )


# ==============================
# Models - litellm response model
# ==============================


class LitellmResponseModelConfig(LitellmModelConfig):
    pass


class LitellmResponseModel(LitellmModel):
    def __init__(self, *, config_class: Any = LitellmResponseModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        result = []
        for msg in messages:
            if msg.get("object") == "response":
                for item in msg.get("output", []):
                    result.append({k: v for k, v in item.items() if k != "extra"})
            else:
                result.append({k: v for k, v in msg.items() if k != "extra"})
        return result

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.responses(
                model=self.config.model_name,
                input=messages,
                tools=[BASH_TOOL_RESPONSE_API],
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(
            logger=logging.getLogger("litellm_response_model"),
            abort_exceptions=self.abort_exceptions,
        ):
            with attempt:
                response = self._query(
                    self._prepare_messages_for_api(messages), **kwargs
                )
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = (
            response.model_dump() if hasattr(response, "model_dump") else dict(response)
        )
        message["extra"] = {
            "actions": self._parse_actions(response),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _parse_actions(self, response) -> list[dict]:
        return parse_toolcall_actions_response(
            getattr(response, "output", []),
            format_error_template=self.config.format_error_template,
        )

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )


# ==============================
# Models - openrouter model
# ==============================

import requests


class OpenRouterModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    set_cache_control: Literal["default_end"] | None = None
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv(
        "MSWEA_COST_TRACKING", "default"
    )
    format_error_template: str = "{{ error }}"
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""


class OpenRouterAPIError(Exception):
    pass


class OpenRouterAuthenticationError(Exception):
    pass


class OpenRouterRateLimitError(Exception):
    pass


class _DictToObj:
    def __init__(self, d: dict):
        self._d = d
        self.id = d.get("id")
        self.function = _DictToObj(d.get("function", {})) if "function" in d else None
        self.name = d.get("name")
        self.arguments = d.get("arguments")


class OpenRouterModel:
    abort_exceptions: list[type[Exception]] = [
        OpenRouterAuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, **kwargs):
        self.config = OpenRouterModelConfig(**kwargs)
        self._api_url = "https://openrouter.ai/api/v1/chat/completions"
        self._api_key = os.getenv("OPENROUTER_API_KEY", "")

    def _query(self, messages: list[dict[str, str]], **kwargs):
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "tools": [BASH_TOOL],
            "usage": {"include": True},
            **(self.config.model_kwargs | kwargs),
        }

        try:
            response = requests.post(
                self._api_url, headers=headers, data=json.dumps(payload), timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                error_msg = "Authentication failed. You can permanently set your API key with `mini-extra config set OPENROUTER_API_KEY YOUR_KEY`."
                raise OpenRouterAuthenticationError(error_msg) from e
            elif response.status_code == 429:
                raise OpenRouterRateLimitError("Rate limit exceeded") from e
            else:
                raise OpenRouterAPIError(
                    f"HTTP {response.status_code}: {response.text}"
                ) from e
        except requests.exceptions.RequestException as e:
            raise OpenRouterAPIError(f"Request failed: {e}") from e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(
            logger=logging.getLogger("openrouter_model"),
            abort_exceptions=self.abort_exceptions,
        ):
            with attempt:
                response = self._query(
                    self._prepare_messages_for_api(messages), **kwargs
                )
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = dict(response["choices"][0]["message"])
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response,
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _calculate_cost(self, response) -> dict[str, float]:
        usage = response.get("usage", {})
        cost = usage.get("cost", 0.0)
        if cost <= 0.0 and self.config.cost_tracking != "ignore_errors":
            raise RuntimeError(
                f"No valid cost information available from OpenRouter API for model {self.config.model_name}: "
                f"Usage {usage}, cost {cost}. Cost must be > 0.0. Set cost_tracking: 'ignore_errors' in your config file or "
                "export MSWEA_COST_TRACKING='ignore_errors' to ignore cost tracking errors "
                "(for example for free/local models), more information at https://klieret.short.gy/mini-local-models "
                "for more details. Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
            )
        return {"cost": cost}

    def _parse_actions(self, response: dict) -> list[dict]:
        tool_calls = response["choices"][0]["message"].get("tool_calls") or []
        tool_calls = [_DictToObj(tc) for tc in tool_calls]
        return parse_toolcall_actions(
            tool_calls, format_error_template=self.config.format_error_template
        )

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


# ==============================
# Models - openrouter textbased model
# ==============================


class OpenRouterTextbasedModelConfig(OpenRouterModelConfig):
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions."


class OpenRouterTextbasedModel(OpenRouterModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = OpenRouterTextbasedModelConfig(**kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "usage": {"include": True},
            **(self.config.model_kwargs | kwargs),
        }

        try:
            response = requests.post(
                self._api_url, headers=headers, data=json.dumps(payload), timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                error_msg = "Authentication failed. You can permanently set your API key with `mini-extra config set OPENROUTER_API_KEY YOUR_KEY`."
                raise OpenRouterAuthenticationError(error_msg) from e
            elif response.status_code == 429:
                raise OpenRouterRateLimitError("Rate limit exceeded") from e
            else:
                raise OpenRouterAPIError(
                    f"HTTP {response.status_code}: {response.text}"
                ) from e
        except requests.exceptions.RequestException as e:
            raise OpenRouterAPIError(f"Request failed: {e}") from e

    def _parse_actions(self, response: dict) -> list[dict]:
        content = response["choices"][0]["message"]["content"] or ""
        return parse_regex_actions(
            content,
            action_regex=self.config.action_regex,
            format_error_template=self.config.format_error_template,
        )

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )


# ==============================
# Models - openrouter response model
# ==============================


class OpenRouterResponseModelConfig(OpenRouterModelConfig):
    pass


class OpenRouterResponseModel(OpenRouterModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = OpenRouterResponseModelConfig(**kwargs)
        self._api_url = "https://openrouter.ai/api/v1/responses"

    def _query(self, messages: list[dict[str, str]], **kwargs):
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model_name,
            "input": messages,
            "tools": [BASH_TOOL_RESPONSE_API],
            **(self.config.model_kwargs | kwargs),
        }
        try:
            response = requests.post(
                self._api_url, headers=headers, data=json.dumps(payload), timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                error_msg = "Authentication failed. You can permanently set your API key with `mini-extra config set OPENROUTER_API_KEY YOUR_KEY`."
                raise OpenRouterAuthenticationError(error_msg) from e
            elif response.status_code == 429:
                raise OpenRouterRateLimitError("Rate limit exceeded") from e
            else:
                raise OpenRouterAPIError(
                    f"HTTP {response.status_code}: {response.text}"
                ) from e
        except requests.exceptions.RequestException as e:
            raise OpenRouterAPIError(f"Request failed: {e}") from e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        result = []
        for msg in messages:
            if msg.get("object") == "response":
                for item in msg.get("output", []):
                    result.append({k: v for k, v in item.items() if k != "extra"})
            else:
                result.append({k: v for k, v in msg.items() if k != "extra"})
        return result

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(
            logger=logging.getLogger("openrouter_response_model"),
            abort_exceptions=self.abort_exceptions,
        ):
            with attempt:
                response = self._query(
                    self._prepare_messages_for_api(messages), **kwargs
                )
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = dict(response)
        message["extra"] = {
            "actions": self._parse_actions(response),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _parse_actions(self, response: dict) -> list[dict]:
        return parse_toolcall_actions_response(
            response.get("output", []),
            format_error_template=self.config.format_error_template,
        )

    def format_message(self, **kwargs) -> dict:
        role = kwargs.get("role", "user")
        content = kwargs.get("content", "")
        extra = kwargs.get("extra")
        content_items = (
            [{"type": "input_text", "text": content}]
            if isinstance(content, str)
            else content
        )
        msg = {"type": "message", "role": role, "content": content_items}
        if extra:
            msg["extra"] = extra
        return msg

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )


# ==============================
# Models - portkey model
# ==============================

try:
    from portkey_ai import Portkey
except ImportError:
    Portkey = None


class PortkeyModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    provider: str = ""
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    litellm_model_name_override: str = ""
    set_cache_control: Literal["default_end"] | None = None
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv(
        "MSWEA_COST_TRACKING", "default"
    )
    format_error_template: str = "{{ error }}"
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""


class PortkeyModel:
    abort_exceptions: list[type[Exception]] = [KeyboardInterrupt, TypeError, ValueError]

    def __init__(self, *, config_class: type = PortkeyModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        if (
            self.config.litellm_model_registry
            and Path(self.config.litellm_model_registry).is_file()
        ):
            litellm.utils.register_model(
                json.loads(Path(self.config.litellm_model_registry).read_text())
            )

        self._api_key = os.getenv("PORTKEY_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Portkey API key is required. Set it via the "
                "PORTKEY_API_KEY environment variable. You can permanently set it with "
                "`mini-extra config set PORTKEY_API_KEY YOUR_KEY`."
            )

        virtual_key = os.getenv("PORTKEY_VIRTUAL_KEY")
        client_kwargs = {"api_key": self._api_key}
        if virtual_key:
            client_kwargs["virtual_key"] = virtual_key
        elif self.config.provider:
            client_kwargs["provider"] = self.config.provider

        if Portkey is None:
            raise ImportError(
                "The portkey-ai package is required to use PortkeyModel. Please install it with: pip install portkey-ai"
            )
        self.client = Portkey(**client_kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        return self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            tools=[BASH_TOOL],
            **(self.config.model_kwargs | kwargs),
        )

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(
            logger=logging.getLogger("portkey_model"),
            abort_exceptions=self.abort_exceptions,
        ):
            with attempt:
                response = self._query(
                    self._prepare_messages_for_api(messages), **kwargs
                )
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _parse_actions(self, response) -> list[dict]:
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(
            tool_calls, format_error_template=self.config.format_error_template
        )

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }

    def _calculate_cost(self, response) -> dict[str, float]:
        response_for_cost_calc = response.model_copy()
        if self.config.litellm_model_name_override:
            if response_for_cost_calc.model:
                response_for_cost_calc.model = self.config.litellm_model_name_override
        prompt_tokens = response_for_cost_calc.usage.prompt_tokens
        if prompt_tokens is None:
            logging.getLogger("portkey_model").warning(
                f"Prompt tokens are None for model {self.config.model_name}. Setting to 0. Full response: {response_for_cost_calc.model_dump()}"
            )
            prompt_tokens = 0
        total_tokens = response_for_cost_calc.usage.total_tokens
        completion_tokens = response_for_cost_calc.usage.completion_tokens
        if completion_tokens is None:
            logging.getLogger("portkey_model").warning(
                f"Completion tokens are None for model {self.config.model_name}. Setting to 0. Full response: {response_for_cost_calc.model_dump()}"
            )
            completion_tokens = 0
        if total_tokens - prompt_tokens - completion_tokens != 0:
            logging.getLogger("portkey_model").warning(
                f"WARNING: Total tokens - prompt tokens - completion tokens != 0: {response_for_cost_calc.model_dump()}."
                " This is probably a portkey bug or incompatibility with litellm cost tracking. "
                "Setting prompt tokens based on total tokens and completion tokens. You might want to double check your costs. "
                f"Full response: {response_for_cost_calc.model_dump()}"
            )
            response_for_cost_calc.usage.prompt_tokens = (
                total_tokens - completion_tokens
            )
        try:
            cost = litellm.cost_calculator.completion_cost(
                response_for_cost_calc,
                model=self.config.litellm_model_name_override or None,
            )
            assert cost >= 0.0, f"Cost is negative: {cost}"
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name} based on {response_for_cost_calc.model_dump()}: {e}. "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors' to ignore this error. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    "Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logging.getLogger("portkey_model").critical(msg)
                raise RuntimeError(msg) from e
        return {"cost": cost}


# ==============================
# Models - portkey response model
# ==============================


class PortkeyResponseAPIModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    litellm_model_name_override: str = ""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv(
        "MSWEA_COST_TRACKING", "default"
    )
    format_error_template: str = "{{ error }}"
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""


class PortkeyResponseAPIModel:
    abort_exceptions: list[type[Exception]] = [KeyboardInterrupt, TypeError, ValueError]

    def __init__(self, **kwargs):
        self.config = PortkeyResponseAPIModelConfig(**kwargs)
        if (
            self.config.litellm_model_registry
            and Path(self.config.litellm_model_registry).is_file()
        ):
            litellm.utils.register_model(
                json.loads(Path(self.config.litellm_model_registry).read_text())
            )

        self._api_key = os.getenv("PORTKEY_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Portkey API key is required. Set it via the "
                "PORTKEY_API_KEY environment variable. You can permanently set it with "
                "`mini-extra config set PORTKEY_API_KEY YOUR_KEY`."
            )

        virtual_key = os.getenv("PORTKEY_VIRTUAL_KEY")
        client_kwargs = {"api_key": self._api_key}
        if virtual_key:
            client_kwargs["virtual_key"] = virtual_key

        if Portkey is None:
            raise ImportError(
                "The portkey-ai package is required to use PortkeyResponseAPIModel. Please install it with: pip install portkey-ai"
            )
        self.client = Portkey(**client_kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        return self.client.responses.create(
            model=self.config.model_name,
            input=messages,
            tools=[BASH_TOOL_RESPONSE_API],
            **(self.config.model_kwargs | kwargs),
        )

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        result = []
        for msg in messages:
            if msg.get("object") == "response":
                for item in msg.get("output", []):
                    result.append({k: v for k, v in item.items() if k != "extra"})
            else:
                result.append({k: v for k, v in msg.items() if k != "extra"})
        return result

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(
            logger=logging.getLogger("portkey_response_model"),
            abort_exceptions=self.abort_exceptions,
        ):
            with attempt:
                response = self._query(
                    self._prepare_messages_for_api(messages), **kwargs
                )
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = (
            response.model_dump() if hasattr(response, "model_dump") else dict(response)
        )
        message["extra"] = {
            "actions": self._parse_actions(response),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _parse_actions(self, response) -> list[dict]:
        output = (
            response.output
            if hasattr(response, "output")
            else response.get("output", [])
        )
        return parse_toolcall_actions_response(
            output, format_error_template=self.config.format_error_template
        )

    def _calculate_cost(self, response) -> dict[str, float]:
        try:
            cost = litellm.cost_calculator.completion_cost(
                response,
                model=self.config.litellm_model_name_override or self.config.model_name,
            )
            assert cost > 0.0, f"Cost is not positive: {cost}"
        except Exception as e:
            if self.config.cost_tracking != "ignore_errors":
                raise RuntimeError(
                    f"Error calculating cost for model {self.config.model_name}: {e}. "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors' to ignore this error. "
                ) from e
            cost = 0.0
        return {"cost": cost}

    def format_message(self, **kwargs) -> dict:
        role = kwargs.get("role", "user")
        content = kwargs.get("content", "")
        extra = kwargs.get("extra")
        content_items = (
            [{"type": "input_text", "text": content}]
            if isinstance(content, str)
            else content
        )
        msg = {"type": "message", "role": role, "content": content_items}
        if extra:
            msg["extra"] = extra
        return msg

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


# ==============================
# Models - requesty model
# ==============================


class RequestyModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    set_cache_control: Literal["default_end"] | None = None
    format_error_template: str = "{{ error }}"
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""


class RequestyAPIError(Exception):
    pass


class RequestyAuthenticationError(Exception):
    pass


class RequestyRateLimitError(Exception):
    pass


class RequestyModel:
    abort_exceptions: list[type[Exception]] = [
        RequestyAuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, **kwargs):
        self.config = RequestyModelConfig(**kwargs)
        self._api_url = "https://router.requesty.ai/v1/chat/completions"
        self._api_key = os.getenv("REQUESTY_API_KEY", "")

    def _query(self, messages: list[dict[str, str]], **kwargs):
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/SWE-agent/mini-swe-agent",
            "X-Title": "mini-swe-agent",
        }

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "tools": [BASH_TOOL],
            **(self.config.model_kwargs | kwargs),
        }

        try:
            response = requests.post(
                self._api_url, headers=headers, data=json.dumps(payload), timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                error_msg = "Authentication failed. You can permanently set your API key with `mini-extra config set REQUESTY_API_KEY YOUR_KEY`."
                raise RequestyAuthenticationError(error_msg) from e
            elif response.status_code == 429:
                raise RequestyRateLimitError("Rate limit exceeded") from e
            else:
                raise RequestyAPIError(
                    f"HTTP {response.status_code}: {response.text}"
                ) from e
        except requests.exceptions.RequestException as e:
            raise RequestyAPIError(f"Request failed: {e}") from e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(
            logger=logging.getLogger("requesty_model"),
            abort_exceptions=self.abort_exceptions,
        ):
            with attempt:
                response = self._query(
                    self._prepare_messages_for_api(messages), **kwargs
                )
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = dict(response["choices"][0]["message"])
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response,
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _calculate_cost(self, response) -> dict[str, float]:
        usage = response.get("usage", {})
        cost = usage.get("cost", 0.0)
        if cost == 0.0:
            raise RequestyAPIError(
                f"No cost information available from Requesty API for model {self.config.model_name}. "
                "Cost tracking is required but not provided by the API response."
            )
        return {"cost": cost}

    def _parse_actions(self, response: dict) -> list[dict]:
        tool_calls = response["choices"][0]["message"].get("tool_calls") or []
        tool_calls = [_DictToObj(tc) for tc in tool_calls]
        return parse_toolcall_actions(
            tool_calls, format_error_template=self.config.format_error_template
        )

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


# ==============================
# Models - test models
# ==============================


def make_output(content: str, actions: list[dict], cost: float = 1.0) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "extra": {"actions": actions, "cost": cost, "timestamp": time.time()},
    }


def make_toolcall_output(
    content: str | None, tool_calls: list[dict], actions: list[dict]
) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
        "extra": {"actions": actions, "cost": 1.0, "timestamp": time.time()},
    }


def make_response_api_output(content: str | None, actions: list[dict]) -> dict:
    output_items = []
    if content:
        output_items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        )
    for action in actions:
        output_items.append(
            {
                "type": "function_call",
                "call_id": action["tool_call_id"],
                "name": "bash",
                "arguments": f'{{"command": "{action["command"]}"}}',
            }
        )
    return {
        "object": "response",
        "output": output_items,
        "extra": {"actions": actions, "cost": 1.0, "timestamp": time.time()},
    }


def _process_test_actions(actions: list[dict]) -> bool:
    for action in actions:
        if "raise" in action:
            raise action["raise"]
        cmd = action.get("command", "")
        if cmd.startswith("/sleep "):
            time.sleep(float(cmd.split("/sleep ")[1]))
            return True
        if cmd.startswith("/warning"):
            logging.warning(cmd.split("/warning")[1])
            return True
    return False


class DeterministicModelConfig(BaseModel):
    outputs: list[dict]
    model_name: str = "deterministic"
    cost_per_call: float = 1.0
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""


class DeterministicModel:
    def __init__(self, **kwargs):
        self.config = DeterministicModelConfig(**kwargs)
        self.current_index = -1

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self.current_index += 1
        output = self.config.outputs[self.current_index]
        if _process_test_actions(output.get("extra", {}).get("actions", [])):
            return self.query(messages, **kwargs)
        GLOBAL_MODEL_STATS.add(self.config.cost_per_call)
        return output

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


class DeterministicToolcallModelConfig(BaseModel):
    outputs: list[dict]
    model_name: str = "deterministic_toolcall"
    cost_per_call: float = 1.0
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""


class DeterministicToolcallModel:
    def __init__(self, **kwargs):
        self.config = DeterministicToolcallModelConfig(**kwargs)
        self.current_index = -1

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self.current_index += 1
        output = self.config.outputs[self.current_index]
        if _process_test_actions(output.get("extra", {}).get("actions", [])):
            return self.query(messages, **kwargs)
        GLOBAL_MODEL_STATS.add(self.config.cost_per_call)
        return output

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


class DeterministicResponseAPIToolcallModelConfig(BaseModel):
    outputs: list[dict]
    model_name: str = "deterministic_response_api_toolcall"
    cost_per_call: float = 1.0
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""


class DeterministicResponseAPIToolcallModel:
    def __init__(self, **kwargs):
        self.config = DeterministicResponseAPIToolcallModelConfig(**kwargs)
        self.current_index = -1

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self.current_index += 1
        output = self.config.outputs[self.current_index]
        if _process_test_actions(output.get("extra", {}).get("actions", [])):
            return self.query(messages, **kwargs)
        GLOBAL_MODEL_STATS.add(self.config.cost_per_call)
        return output

    def format_message(self, **kwargs) -> dict:
        role = kwargs.get("role", "user")
        content = kwargs.get("content", "")
        extra = kwargs.get("extra")
        content_items = (
            [{"type": "input_text", "text": content}]
            if isinstance(content, str)
            else content
        )
        msg: dict = {"type": "message", "role": role, "content": content_items}
        if extra:
            msg["extra"] = extra
        return msg

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages_response(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


# ==============================
# Config
# ==============================

import yaml


# Default configurations
DEFAULT_AGENT_CONFIG = {
    "agent": {
        "system_template": "You are a helpful assistant that can execute commands on the local machine. When you are done, you must submit your final answer. To submit, you must print 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' followed by your final answer.",
        "instance_template": "Complete the following task: {{task}}",
        "step_limit": 10,
        "cost_limit": 3.0,
        "output_path": None,
    }
}


def _key_value_spec_to_nested_dict(config_spec: str) -> dict:
    key, value = config_spec.split("=", 1)
    try:
        value = json.loads(value)
    except json.JSONDecodeError:
        pass
    keys = key.split(".")
    result = {}
    current = result
    for k in keys[:-1]:
        current[k] = {}
        current = current[k]
    current[keys[-1]] = value
    return result


def get_config_from_spec(config_spec: str | Path) -> dict:
    if isinstance(config_spec, str) and "=" in config_spec:
        return _key_value_spec_to_nested_dict(config_spec)
    # Return default config if no file is specified
    return DEFAULT_AGENT_CONFIG


# ==============================
# Run - config utility
# ==============================

from dotenv import set_key, unset_key
from prompt_toolkit import prompt
from rich.rule import Rule
from typer import Argument, Typer


def configure_if_first_time():
    # No-op since we've removed environment variable configuration
    pass


# ==============================
# Run - main mini command
# ==============================

import typer

app = Typer(rich_markup_mode="rich")

DEFAULT_OUTPUT_FILE = package_dir / "last_mini_run.traj.json"


_HELP_TEXT = """Run mini-SWE-agent in your local environment.

[not dim]
More information about the usage: [bold green]https://mini-sweagent.com/latest/usage/mini/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c mini.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c mini.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml agent.mode=yolo[/bold green]
"""


@app.command(help=_HELP_TEXT)
def main(
    model_name: str | None = typer.Option(
        None,
        "-m",
        "--model",
        help="Model to use",
    ),
    model_class: str | None = typer.Option(
        None,
        "--model-class",
        help="Model class to use (e.g., 'litellm' or 'minisweagent.models.litellm_model.LitellmModel')",
        rich_help_panel="Advanced",
    ),
    agent_class: str | None = typer.Option(
        None,
        "--agent-class",
        help="Agent class to use (e.g., 'interactive' or 'minisweagent.agents.interactive.InteractiveAgent')",
        rich_help_panel="Advanced",
    ),
    environment_class: str | None = typer.Option(
        None,
        "--environment-class",
        help="Environment class to use (e.g., 'local' or 'minisweagent.environments.local.LocalEnvironment')",
        rich_help_panel="Advanced",
    ),
    task: str | None = typer.Option(
        None, "-t", "--task", help="Task/problem statement", show_default=False
    ),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(
        None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."
    ),
    config_spec: list[str] = typer.Option(
        [], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT
    ),
    output: Path | None = typer.Option(
        DEFAULT_OUTPUT_FILE, "-o", "--output", help="Output trajectory file"
    ),
    exit_immediately: bool = typer.Option(
        False,
        "--exit-immediately",
        help="Exit immediately when the agent wants to finish instead of prompting.",
        rich_help_panel="Advanced",
    ),
) -> Any:
    configure_if_first_time()

    console = Console(highlight=False)
    console.print(
        f"Building agent config from specs: [bold green]{config_spec}[/bold green]"
    )
    configs = [DEFAULT_AGENT_CONFIG]
    for spec in config_spec:
        configs.append(get_config_from_spec(spec))
    configs.append(
        {
            "run": {
                "task": task or UNSET,
            },
            "agent": {
                "agent_class": agent_class or UNSET,
                "mode": "yolo" if yolo else UNSET,
                "cost_limit": cost_limit or UNSET,
                "confirm_exit": False if exit_immediately else UNSET,
                "output_path": output or UNSET,
            },
            "model": {
                "model_class": model_class or UNSET,
                "model_name": model_name or UNSET,
            },
            "environment": {
                "environment_class": environment_class or UNSET,
            },
        }
    )
    config = recursive_merge(*configs)

    if (run_task := config.get("run", {}).get("task", UNSET)) is UNSET:
        console.print("[bold yellow]What do you want to do?")
        run_task = _multiline_prompt()
        console.print("[bold green]Got that, thanks![/bold green]")

    model = get_model(config=config.get("model", {}))
    env = get_environment(config.get("environment", {}), default_type="local")
    agent = get_agent(model, env, config.get("agent", {}), default_type="interactive")
    agent.run(run_task)
    if output_path := config.get("agent", {}).get("output_path"):
        console.print(f"Saved trajectory to [bold green]'{output_path}'[/bold green]")
    return agent


# ==============================
# Run - hello world example
# ==============================

hello_world_app = Typer()


@hello_world_app.command()
def hello_world(
    task: str = typer.Option(
        ...,
        "-t",
        "--task",
        help="Task/problem statement",
        show_default=False,
        prompt=True,
    ),
    model_name: str = typer.Option(
        None,
        "-m",
        "--model",
        help="Model name",
        prompt="What model do you want to use?",
    ),
) -> DefaultAgent:
    logging.basicConfig(level=logging.DEBUG)
    agent = DefaultAgent(
        LitellmModel(model_name=model_name),
        LocalEnvironment(),
        **DEFAULT_AGENT_CONFIG["agent"],
    )
    agent.run(task)
    return agent


# ==============================
# Entry Point
# ==============================

if __name__ == "__main__":
    app()
