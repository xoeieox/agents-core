import sys
from agents_core.room_paths import _emit_sh, _emit_rust

if "--emit-sh" in sys.argv:
    print(_emit_sh(), end="")
elif "--emit-rust" in sys.argv:
    print(_emit_rust(), end="")
else:
    print("Usage: python -m agents_core.room_paths [--emit-sh | --emit-rust]", file=sys.stderr)
    sys.exit(1)
