import subprocess
import tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
origin = tmp / "origin.git"
origin.mkdir()
subprocess.run(["git", "init", "--bare", str(origin)], check=True, capture_output=True)

seed = tmp / "seed"
seed.mkdir()
subprocess.run(["git", "-C", str(seed), "init", "-b", "main"], check=True, capture_output=True)
subprocess.run(["git", "-C", str(seed), "config", "user.name", "scratch"], check=True, capture_output=True)
subprocess.run(["git", "-C", str(seed), "config", "user.email", "scratch@example.com"], check=True, capture_output=True)
(seed / "f.py").write_text("VALUE = 0\n")
tests_dir = seed / "tests"
tests_dir.mkdir()
(tests_dir / "test_fast.py").write_text("def test_ok():\n    assert True\n")
subprocess.run(["git", "-C", str(seed), "add", "-A"], check=True, capture_output=True)
subprocess.run(["git", "-C", str(seed), "commit", "-m", "seed"], check=True, capture_output=True)
subprocess.run(["git", "-C", str(seed), "remote", "add", "origin", str(origin)], check=True, capture_output=True)
subprocess.run(["git", "-C", str(seed), "push", "origin", "main"], check=True, capture_output=True)

clone = tmp / "clone"
subprocess.run(["git", "clone", str(origin), str(clone)], check=True, capture_output=True)
subprocess.run(["git", "-C", str(clone), "config", "user.name", "scratch"], check=True, capture_output=True)
subprocess.run(["git", "-C", str(clone), "config", "user.email", "scratch@example.com"], check=True, capture_output=True)

# Now the failing sequence:
(clone / "uv.lock").write_text("[[package]]\nname = 'scratch'\n")
r = subprocess.run(["git", "-C", str(clone), "add", "uv.lock"], capture_output=True, text=True)
print("add rc:", r.returncode, r.stderr)
r = subprocess.run(["git", "-C", str(clone), "commit", "-qm", "add uv.lock"], capture_output=True, text=True)
print("commit rc:", r.returncode, r.stderr)
r = subprocess.run(["git", "-C", str(clone), "push", "-q", "origin", "main"], capture_output=True, text=True)
print("push rc:", r.returncode, "stdout:", r.stdout, "stderr:", r.stderr)
