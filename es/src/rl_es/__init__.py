import numpy as np

# NumPy 2.0 removed several legacy scalar aliases (np.float_, np.int_, ...). Some
# upstream Gymnasium environments still reference np.float_, so we add the alias
# back when needed to keep those environments running until they update.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
