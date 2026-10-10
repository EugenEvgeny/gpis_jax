"""Numerical regression suite uses float64 to check derivative formulas tightly."""
import jax

jax.config.update('jax_enable_x64', True)
