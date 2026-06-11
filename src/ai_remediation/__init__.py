"""
InfraGraph AI — AI remediation layer.

Provides:
  generate_resolution_plan          — unified entry point (local + enterprise)
  build_remediation_prompt         — construct a Qwen3 chat prompt from RCA context
  generate_remediation_with_qwen   — call vLLM/OpenAI-compatible endpoint
  check_vllm_available             — health-check the local vLLM server
  generate_template_remediation    — deterministic template mode (not model-generated)
  make_remediation_input           — normalise remediation input
  empty_remediation_output         — empty output skeleton
  make_remediation_output          — normalise remediation output
"""
from .prompt_builder  import build_remediation_prompt
from .qwen_client     import (
    generate_remediation_with_qwen,
    check_vllm_available,
    generate_resolution_plan,
)
from .template_mode   import generate_template_remediation
from .response_schema import (
    make_remediation_input,
    empty_remediation_output,
    make_remediation_output,
)

__all__ = [
    "generate_resolution_plan",
    "build_remediation_prompt",
    "generate_remediation_with_qwen",
    "check_vllm_available",
    "generate_template_remediation",
    "make_remediation_input",
    "empty_remediation_output",
    "make_remediation_output",
]
