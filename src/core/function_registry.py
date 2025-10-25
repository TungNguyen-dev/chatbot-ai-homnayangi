"""
Central registry for discovering, managing, and dispatching LLM tool functions.

Automatically scans packages for tool modules, extracts OpenAI-style DEFINITION
metadata, registers callable handlers, and dispatches tool calls from model outputs.
"""

import importlib
import json
import logging
import pkgutil
from types import ModuleType
from typing import Dict, Generator, Optional, Any, List, Callable, Iterable

logger = logging.getLogger(__name__)


class FunctionRegistry:
    """
    Handles parsing of tool calls, business logic, and LLM prompt utilities.
    Also, responsible for discovering tool function handlers and OpenAI tool
    definitions (DEFINITION) exposed by function modules.
    """

    PREFERRED_PACKAGE = "src.functions"
    LEGACY_PACKAGE = "src.core.functions"

    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client
        self.function_handlers: Dict[str, Callable[[Any, dict], str]] = {}
        self.tool_definitions: List[dict] = []
        self.reload_function_handlers()

    # --------------------------------------------------------
    # Module Discovery
    # --------------------------------------------------------

    @staticmethod
    def _iter_modules_in_package(package_name: str) -> Generator[ModuleType, None, None]:
        """Yield imported modules for all children within a package, safely."""
        try:
            package = importlib.import_module(package_name)
        except ModuleNotFoundError:
            logger.warning("Package '%s' not found.", package_name)
            return
        except ImportError as exc:
            logger.error("Failed to import package '%s': %s", package_name, exc)
            return

        package_path = getattr(package, "__path__", None)
        if not package_path:
            logger.debug("Package '%s' has no __path__, skipping.", package_name)
            return

        for module_info in pkgutil.iter_modules(package_path):
            if module_info.name.startswith("_"):
                continue

            full_name = f"{package_name}.{module_info.name}"
            try:
                yield importlib.import_module(full_name)
            except ModuleNotFoundError:
                logger.warning("Module '%s' not found.", full_name)
            except ImportError as exc:
                logger.error("Error importing module '%s': %s", full_name, exc)

    # --------------------------------------------------------
    # Function Handler Loading
    # --------------------------------------------------------

    def _load_function_handlers(self) -> Dict[str, Callable]:
        """Auto-discover function handler modules and build handler map."""
        handlers: Dict[str, Callable] = {}

        def load_from_package(package_name: str, override_existing: bool) -> None:
            for mod in self._iter_modules_in_package(package_name):
                try:
                    definition = getattr(mod, "DEFINITION", None)
                    handle = getattr(mod, "handle", None)
                except AttributeError:
                    logger.debug("Module '%s' missing attributes.", mod.__name__)
                    continue

                if not callable(handle):
                    logger.debug("Module '%s' has no callable 'handle'.", mod.__name__)
                    continue

                func_name = (
                                    isinstance(definition, dict)
                                    and definition.get("function", {}).get("name")
                            ) or mod.__name__.rsplit(".", 1)[-1]

                if func_name and (override_existing or func_name not in handlers):
                    handlers[func_name] = handle
                    logger.debug("Registered handler '%s' from %s.", func_name, mod.__name__)

        # Preferred package has higher priority
        load_from_package(self.PREFERRED_PACKAGE, override_existing=True)
        load_from_package(self.LEGACY_PACKAGE, override_existing=False)

        return handlers

    # --------------------------------------------------------
    # Tool Definition Collection
    # --------------------------------------------------------

    def _collect_tool_definitions(self) -> List[dict]:
        """Collect DEFINITION dicts from discovered modules."""
        definitions: List[dict] = []

        def extract_definitions(package_name: str) -> None:
            for mod in self._iter_modules_in_package(package_name):
                definition = getattr(mod, "DEFINITION", None)
                if not isinstance(definition, dict):
                    continue

                func = definition.get("function")
                if isinstance(func, dict) and isinstance(func.get("name"), str):
                    definitions.append(definition)

        extract_definitions(self.PREFERRED_PACKAGE)
        extract_definitions(self.LEGACY_PACKAGE)

        return definitions

    # --------------------------------------------------------
    # Reload
    # --------------------------------------------------------

    def reload_function_handlers(self) -> None:
        """Reload function handlers and tool definitions by rescanning packages."""
        self.function_handlers = self._load_function_handlers()
        self.tool_definitions = self._collect_tool_definitions()

    # --------------------------------------------------------
    # Stream Handler
    # --------------------------------------------------------

    def handle_stream(self, stream) -> Generator[str, None, None]:
        """
        Parse stream chunks and handle both text and function calls.

        Collects model-generated text incrementally and detects any function calls
        returned by the model. After streaming ends, dispatches the function call.
        """
        buffer_parts: List[str] = []
        function_name: Optional[str] = None

        for chunk in stream:
            delta = chunk.choices[0].delta

            # Yield normal content chunks immediately
            if getattr(delta, "content", None):
                yield delta.content
                continue

            # Handle tool (function) call events
            if getattr(delta, "tool_calls", None):
                tool_call = delta.tool_calls[0]
                func = getattr(tool_call, "function", None)
                if not func:
                    continue

                if getattr(func, "name", None):
                    function_name = func.name

                if getattr(func, "arguments", None):
                    buffer_parts.append(func.arguments)

        # Dispatch the accumulated function call (if any)
        if function_name:
            yield from self.dispatch(function_name, "".join(buffer_parts))

    # --------------------------------------------------------
    # Function Dispatch
    # --------------------------------------------------------

    def dispatch(self, function_name: str, arguments_buffer: str) -> Generator[str, None, None]:
        """
        Dispatch a function call to the corresponding handler.

        Parses JSON arguments, calls the correct handler, and yields the result.
        """
        logger.info("Dispatching function call: %s", function_name)

        handler = self.function_handlers.get(function_name)
        if handler is None:
            msg = f"⚠️ Function '{function_name}' not found among registered handlers."
            logger.warning(msg)
            yield msg
            return

        # Parse arguments_buffer robustly because streamed tool arguments can arrive
        # as multiple concatenated JSON fragments or with trailing data.
        def _parse_args(buffer: str) -> Dict[str, Any]:
            s = (buffer or "").strip()
            if not s:
                return {}
            # First, try the fast path
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
                raise ValueError("Parsed arguments must be a JSON object.")
            except Exception:
                # Fallback: incrementally decode and take the last complete JSON object
                decoder = json.JSONDecoder()
                idx = 0
                last_obj: Optional[Any] = None
                # Allow comma separators between objects just in case
                while idx < len(s):
                    # Skip whitespace and commas
                    while idx < len(s) and s[idx] in " \t\r\n,":
                        idx += 1
                    if idx >= len(s):
                        break
                    try:
                        obj, end = decoder.raw_decode(s, idx)
                        last_obj = obj
                        idx = end
                    except json.JSONDecodeError:
                        # If we cannot decode further, stop
                        break
                if isinstance(last_obj, dict):
                    return last_obj
                raise ValueError("Could not extract a JSON object from streamed arguments.")

        try:
            args = _parse_args(arguments_buffer)
        except Exception as exc:
            msg = f"❌ Failed to parse arguments for '{function_name}': {exc}"
            logger.error(msg)
            yield msg
            return

        try:
            logger.debug("Executing handler '%s' with args: %s", function_name, args)
            result: Any = handler(self.llm_client, args)
        except Exception as exc:
            msg = f"❌ Error executing function '{function_name}': {exc}"
            logger.exception(msg)
            yield msg
            return

        # Normalize result and yield appropriately
        if result is None:
            yield f"✅ Function '{function_name}' executed successfully (no return value)."
        elif isinstance(result, str):
            yield result
        elif isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
            for chunk in result:
                yield str(chunk)
        else:
            yield str(result)
