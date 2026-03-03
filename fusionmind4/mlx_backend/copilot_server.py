"""
Copilot Server — Optional vllm-mlx Integration for PF8
========================================================

Serves a local LLM via vllm-mlx for the Causal Copilot (PF8).
The LLM interprets causal graphs, generates hypotheses, and
answers operator queries about plasma causal relationships.

This is OPTIONAL — FusionMind core runs without it.

Setup:
    pip install vllm-mlx
    # or
    pip install git+https://github.com/waybarrios/vllm-mlx.git

Usage:
    from fusionmind4.mlx_backend.copilot_server import CopilotServer

    server = CopilotServer(model="mlx-community/Llama-3.2-3B-Instruct-4bit")
    server.start(port=8000)

    # Query from Copilot
    answer = server.query("What happens if I increase NBI power by 2MW?",
                          causal_context=copilot.build_context(scm, dag))

Author: Dr. Mladen Mester, March 2026
"""

import json
import subprocess
import sys
import time
from typing import Dict, Optional

VLLM_MLX_AVAILABLE = False
try:
    import importlib
    importlib.import_module("vllm_mlx")
    VLLM_MLX_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass


# Default models ranked by quality/size tradeoff for Mac
RECOMMENDED_MODELS = {
    "small":  "mlx-community/Llama-3.2-3B-Instruct-4bit",    # ~2GB, fast
    "medium": "mlx-community/Qwen3-8B-Instruct-4bit",         # ~5GB, good reasoning
    "large":  "mlx-community/Llama-3.1-70B-Instruct-4bit",    # ~40GB, M3/M4 Ultra
}


class CopilotServer:
    """
    Manages a local vllm-mlx server for the Causal Copilot.

    The server runs as a subprocess, exposing an OpenAI-compatible API.
    The Copilot module queries it with causal context injected.
    """

    def __init__(self, model: str = None, port: int = 8321):
        """
        Args:
            model: HuggingFace model ID (mlx-community format)
            port: Server port
        """
        self.model = model or RECOMMENDED_MODELS["small"]
        self.port = port
        self._process = None

    def start(self, continuous_batching: bool = False):
        """Start vllm-mlx server as subprocess."""
        if not VLLM_MLX_AVAILABLE:
            raise ImportError(
                "vllm-mlx is not installed.\n"
                "Install: pip install git+https://github.com/waybarrios/vllm-mlx.git\n"
                "This is optional — FusionMind core works without it."
            )

        cmd = [
            sys.executable, "-m", "vllm_mlx", "serve",
            self.model, "--port", str(self.port),
        ]
        if continuous_batching:
            cmd.append("--continuous-batching")

        print(f"Starting Copilot LLM server: {self.model}")
        print(f"  Port: {self.port}")
        print(f"  Endpoint: http://localhost:{self.port}/v1/chat/completions")

        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Wait for server to be ready
        self._wait_for_ready()
        print("  ✓ Server ready")

    def _wait_for_ready(self, timeout: int = 120):
        """Wait for server health check."""
        import urllib.request
        url = f"http://localhost:{self.port}/health"
        start = time.time()
        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(url, timeout=2)
                return
            except Exception:
                time.sleep(2)
        raise TimeoutError(f"vllm-mlx server didn't start in {timeout}s")

    def query(self, question: str, causal_context: str = "",
              max_tokens: int = 512, temperature: float = 0.3) -> str:
        """
        Query the Copilot LLM with causal context.

        Args:
            question: Operator's question
            causal_context: Injected from CausalContextBuilder (PF8)
            max_tokens: Max response length
            temperature: Sampling temperature (lower = more focused)

        Returns:
            LLM response string
        """
        import urllib.request

        system_prompt = (
            "You are FusionMind Causal Copilot, an expert AI assistant for "
            "tokamak plasma control. You use causal inference (Pearl's do-calculus) "
            "to answer questions about plasma behavior. Base your answers on the "
            "provided causal graph and structural equations. Be precise about "
            "causal vs correlational claims."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if causal_context:
            messages.append({
                "role": "user",
                "content": f"[CAUSAL CONTEXT]\n{causal_context}\n[END CONTEXT]"
            })
            messages.append({
                "role": "assistant",
                "content": "I've loaded the causal graph and structural equations. "
                           "I'll use do-calculus reasoning for my answers."
            })
        messages.append({"role": "user", "content": question})

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()

        req = urllib.request.Request(
            f"http://localhost:{self.port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Copilot LLM error: {e}]"

    def stop(self):
        """Stop the server."""
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=10)
            self._process = None
            print("Copilot server stopped.")

    def __del__(self):
        self.stop()


def check_vllm_mlx_status() -> Dict[str, bool]:
    """Check vllm-mlx availability and system compatibility."""
    import platform
    status = {
        "vllm_mlx_installed": VLLM_MLX_AVAILABLE,
        "macos": platform.system() == "Darwin",
        "apple_silicon": platform.machine() == "arm64",
    }

    if status["macos"]:
        # Check macOS version
        ver = platform.mac_ver()[0]
        status["macos_version"] = ver
        major = int(ver.split(".")[0]) if ver else 0
        status["macos_compatible"] = major >= 13

    return status
