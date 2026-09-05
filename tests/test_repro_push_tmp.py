import subprocess


def test_repro_push(tmp_path):
    tmp = tmp_path
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

    (clone / "uv.lock").write_text("[[package]]\nname = 'scratch'\n")
    subprocess.run(["git", "-C", str(clone), "add", "uv.lock"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "commit", "-qm", "add uv.lock"], check=True, capture_output=True)

    # Inspect the clone's branch state:
    r = subprocess.run(["git", "-C", str(clone), "branch", "-a"], capture_output=True, text=True)
    branches = r.stdout
    r = subprocess.run(["git", "-C", str(clone), "status"], capture_output=True, text=True)
    status = r.stdout
    r = subprocess.run(["git", "--version"], capture_output=True, text=True)
    version = r.stdout
    # Force a failure with the diagnostics embedded:
    assert False, f"branches={branches!r} status={status!r} git={version!r}"
