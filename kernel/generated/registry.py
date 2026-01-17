# kernel/generated/registry.py
"""
Generated kernel registry and control flags.

This controls whether generated kernels are allowed to run at all.
Used for:
- global enable / disable
- emergency kill-switch
- debug sessions
"""

# Global switch
_GENERATED_KERNELS_ENABLED: bool = True


def enable_generated_kernels():
    global _GENERATED_KERNELS_ENABLED
    _GENERATED_KERNELS_ENABLED = True


def disable_generated_kernels():
    global _GENERATED_KERNELS_ENABLED
    _GENERATED_KERNELS_ENABLED = False


def generated_kernels_enabled() -> bool:
    return _GENERATED_KERNELS_ENABLED
