#!/usr/bin/env python3
"""Forgejo API helper — shared by all agent scripts that interact with Forgejo.

Hosted on BRIX (203.0.113.10:3000) since the 2026-06 substrate move.

Ownership model (decision 2026-06-02): agent-managed repos live under the
``lapis`` org, operated by the ``conductor`` Forgejo user (org owner). Existing
human repos stay under ``Erah/``. Functions default to ``OWNER`` (Erah, for
backward-compat with code-reviewer / pr-status-sync etc.); pass ``owner="lapis"``
(or any org/user) to target the agent namespace.

Mutating operations emit a provenance record to a human-readable JSONL ledger
(FORGEJO_PROVENANCE_LOG, default /srv/agents/logs/forgejo-repo-ops.jsonl) per the
Lapis agent-operable-backend rule. Destructive ops (delete_repo) require an
explicit confirm=True flag.
"""

import json
import os
import subprocess
from datetime import datetime, timezone

import httpx

FORGEJO_URL = os.environ.get("FORGEJO_URL", "http://203.0.113.10:3000")
FORGEJO_TOKEN = os.environ.get("FORGEJO_TOKEN", "")
OWNER = os.environ.get("FORGEJO_OWNER", "Erah")          # default namespace (human repos)
LAPIS_ORG = os.environ.get("FORGEJO_LAPIS_ORG", "lapis")  # agent-managed namespace
ACTOR = os.environ.get("LAPIS_ACTOR", "conductor")        # provenance actor identity
PROVENANCE_LOG = os.environ.get(
    "FORGEJO_PROVENANCE_LOG", "/srv/agents/logs/forgejo-repo-ops.jsonl"
)
API = f"{FORGEJO_URL}/api/v1"
TIMEOUT = 15


def _headers():
    return {"Authorization": f"token {FORGEJO_TOKEN}", "Accept": "application/json"}


def _owner(owner: str | None) -> str:
    return owner or OWNER


def _provenance(op: str, owner: str, repo: str, detail: dict | None = None,
                ok: bool = True, error: str = "") -> None:
    """Append a provenance record for a mutating op. Fail-safe: never raises."""
    try:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "actor": ACTOR,
            "op": op,
            "owner": owner,
            "repo": repo,
            "detail": detail or {},
            "ok": ok,
        }
        if error:
            rec["error"] = error
        os.makedirs(os.path.dirname(PROVENANCE_LOG) or ".", exist_ok=True)
        with open(PROVENANCE_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Repo lifecycle (create / push / delete)
# ---------------------------------------------------------------------------
def create_repo(repo: str, description: str = "", private: bool = True,
                owner: str | None = None) -> dict:
    """Create a new repository.

    owner=None  -> created under the authenticated user (conductor) namespace.
    owner="lapis" (or any org) -> created in that org (POST /orgs/{org}/repos).
    """
    body = {"name": repo, "description": description, "private": private, "auto_init": False}
    if owner:
        endpoint = f"{API}/orgs/{owner}/repos"
    else:
        endpoint = f"{API}/user/repos"
    try:
        r = httpx.post(endpoint, headers=_headers(), json=body, timeout=TIMEOUT)
        r.raise_for_status()
    except httpx.HTTPError as e:
        _provenance("create_repo", owner or "(self)", repo, {"private": private}, ok=False, error=str(e))
        raise
    _provenance("create_repo", owner or "(self)", repo, {"private": private, "description": description})
    return r.json()


def create_and_push(repo: str, local_path: str, owner: str | None = None,
                    description: str = "", private: bool = True,
                    default_branch: str = "main", remote_name: str = "origin") -> dict:
    """Create a repo (org-aware) and push an existing local git working tree to it.

    The one-shot handoff path: hand conductor a local repo + a name and it lands
    on Forgejo under `owner`. Sets `remote_name` to the token-authed URL and pushes
    all branches + tags. Idempotent-ish: if the remote already exists it is updated.
    """
    info = create_repo(repo, description=description, private=private, owner=owner)
    target = _owner(owner) if owner else _authed_user()
    # token-authed push URL (host without scheme prefix duplication)
    host = FORGEJO_URL.split("://", 1)[-1]
    push_url = f"http://{ACTOR}:{FORGEJO_TOKEN}@{host}/{target}/{repo}.git"
    def _git(*args):
        return subprocess.run(["git", "-C", local_path, *args], capture_output=True, text=True)
    # set/replace remote
    if _git("remote", "get-url", remote_name).returncode == 0:
        _git("remote", "set-url", remote_name, push_url)
    else:
        _git("remote", "add", remote_name, push_url)
    push = _git("push", "-u", remote_name, "--all")
    tags = _git("push", remote_name, "--tags")
    ok = push.returncode == 0
    # scrub token from any stored remote URL — leave a clean (tokenless) origin
    clean_url = f"{FORGEJO_URL}/{target}/{repo}.git"
    _git("remote", "set-url", remote_name, clean_url)
    _provenance("create_and_push", target, repo,
                {"local_path": local_path, "default_branch": default_branch,
                 "push_rc": push.returncode, "tags_rc": tags.returncode},
                ok=ok, error="" if ok else push.stderr.strip()[:300])
    if not ok:
        raise RuntimeError(f"git push failed: {push.stderr.strip()[:300]}")
    return {"repo": info, "pushed": True, "clone_url": clean_url}


def delete_repo(repo: str, owner: str | None = None, confirm: bool = False) -> None:
    """DESTRUCTIVE: delete a repository. Requires confirm=True (human-gated)."""
    target = _owner(owner)
    if not confirm:
        raise PermissionError(
            f"delete_repo({target}/{repo}) refused: destructive op requires confirm=True"
        )
    try:
        r = httpx.delete(f"{API}/repos/{target}/{repo}", headers=_headers(), timeout=TIMEOUT)
        r.raise_for_status()
    except httpx.HTTPError as e:
        _provenance("delete_repo", target, repo, ok=False, error=str(e))
        raise
    _provenance("delete_repo", target, repo, {"confirmed": True})


def list_repos(owner: str | None = None, limit: int = 50) -> list[dict]:
    """List repos under an org/user (read-only)."""
    target = _owner(owner)
    r = httpx.get(f"{API}/orgs/{target}/repos", headers=_headers(),
                  params={"limit": limit}, timeout=TIMEOUT)
    if r.status_code == 404:  # not an org — try user namespace
        r = httpx.get(f"{API}/users/{target}/repos", headers=_headers(),
                      params={"limit": limit}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _authed_user() -> str:
    r = httpx.get(f"{API}/user", headers=_headers(), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["login"]


# ---------------------------------------------------------------------------
# Repo configuration / collaboration
# ---------------------------------------------------------------------------
def add_collaborator(repo: str, username: str, permission: str = "write",
                     owner: str | None = None) -> None:
    """Add a collaborator to a repository."""
    target = _owner(owner)
    r = httpx.put(
        f"{API}/repos/{target}/{repo}/collaborators/{username}",
        headers=_headers(),
        json={"permission": permission},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    _provenance("add_collaborator", target, repo, {"username": username, "permission": permission})


def create_webhook(repo: str, target_url: str, secret: str, events: list[str] | None = None,
                   owner: str | None = None) -> dict:
    """Create a webhook on a repository."""
    target = _owner(owner)
    r = httpx.post(
        f"{API}/repos/{target}/{repo}/hooks",
        headers=_headers(),
        json={
            "type": "forgejo",
            "active": True,
            "config": {"url": target_url, "content_type": "json", "secret": secret},
            "events": events or ["push"],
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    _provenance("create_webhook", target, repo, {"target_url": target_url, "events": events or ["push"]})
    return r.json()


def create_pr(repo: str, title: str, head: str, base: str = "main", body: str = "",
              owner: str | None = None) -> dict:
    """Create a pull request."""
    target = _owner(owner)
    r = httpx.post(
        f"{API}/repos/{target}/{repo}/pulls",
        headers=_headers(),
        json={"title": title, "head": head, "base": base, "body": body},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    _provenance("create_pr", target, repo, {"title": title, "head": head, "base": base})
    return r.json()


def merge_pr(repo: str, pr_number: int, method: str = "merge", owner: str | None = None) -> dict:
    """Merge a pull request."""
    target = _owner(owner)
    r = httpx.post(
        f"{API}/repos/{target}/{repo}/pulls/{pr_number}/merge",
        headers=_headers(),
        json={"Do": method, "delete_branch_after_merge": True},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    _provenance("merge_pr", target, repo, {"pr_number": pr_number, "method": method})
    if r.content:
        return r.json()
    return {"status": "ok"}


def add_comment(repo: str, issue_or_pr: int, body: str, owner: str | None = None) -> dict:
    """Add a comment to an issue or PR."""
    target = _owner(owner)
    r = httpx.post(
        f"{API}/repos/{target}/{repo}/issues/{issue_or_pr}/comments",
        headers=_headers(),
        json={"body": body},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def get_open_prs(repo: str, owner: str | None = None) -> list[dict]:
    """List open pull requests."""
    r = httpx.get(
        f"{API}/repos/{_owner(owner)}/{repo}/pulls",
        headers=_headers(),
        params={"state": "open", "limit": 50},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def get_pr_diff(repo: str, pr_number: int, owner: str | None = None) -> str:
    """Get the diff of a pull request."""
    r = httpx.get(
        f"{API}/repos/{_owner(owner)}/{repo}/pulls/{pr_number}.diff",
        headers=_headers(),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.text


def create_issue(repo: str, title: str, body: str, labels: list[str] | None = None,
                 owner: str | None = None) -> dict:
    """Create an issue on a repository."""
    payload = {"title": title, "body": body}
    if labels:
        # Forgejo expects label IDs, not names — look up or skip
        pass
    r = httpx.post(
        f"{API}/repos/{_owner(owner)}/{repo}/issues",
        headers=_headers(),
        json=payload,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def get_pr(repo: str, pr_number: int, owner: str | None = None) -> dict:
    """Fetch a single pull request (includes mergeable, state, html_url)."""
    r = httpx.get(
        f"{API}/repos/{_owner(owner)}/{repo}/pulls/{pr_number}",
        headers=_headers(),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def get_branch(repo: str, branch: str, owner: str | None = None) -> dict:
    """Fetch branch info; raises httpx.HTTPStatusError (404) if the branch is deleted."""
    r = httpx.get(
        f"{API}/repos/{_owner(owner)}/{repo}/branches/{branch}",
        headers=_headers(),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def close_pr(repo: str, pr_number: int, owner: str | None = None) -> dict:
    """Close a pull request without merging."""
    r = httpx.patch(
        f"{API}/repos/{_owner(owner)}/{repo}/pulls/{pr_number}",
        headers=_headers(),
        json={"state": "closed"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    _provenance("close_pr", _owner(owner), repo, {"pr_number": pr_number})
    return r.json()


def get_pr_comments(repo: str, pr_number: int, owner: str | None = None) -> list[dict]:
    """Fetch comments on a PR (uses issues endpoint, which covers PR comments)."""
    r = httpx.get(
        f"{API}/repos/{_owner(owner)}/{repo}/issues/{pr_number}/comments",
        headers=_headers(),
        params={"limit": 50},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def list_webhooks(repo: str, owner: str | None = None) -> list[dict]:
    """List webhooks configured on a repository."""
    r = httpx.get(
        f"{API}/repos/{_owner(owner)}/{repo}/hooks",
        headers=_headers(),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def update_webhook(repo: str, hook_id: int, events: list[str], owner: str | None = None) -> dict:
    """Update a webhook's event list."""
    r = httpx.patch(
        f"{API}/repos/{_owner(owner)}/{repo}/hooks/{hook_id}",
        headers=_headers(),
        json={"events": events},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    _provenance("update_webhook", _owner(owner), repo, {"hook_id": hook_id, "events": events})
    return r.json()


def set_branch_protection(repo: str, branch: str = "main", required_approvals: int = 0,
                          owner: str | None = None) -> dict:
    """Set branch protection rules."""
    target = _owner(owner)
    r = httpx.post(
        f"{API}/repos/{target}/{repo}/branch_protections",
        headers=_headers(),
        json={
            "branch_name": branch,
            "enable_push": False,
            "enable_push_whitelist": True,
            "push_whitelist_usernames": ["Erah", "conductor"],
            "require_signed_commits": False,
            "required_approvals": required_approvals,
            "block_admin_merge_override": False,
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    _provenance("set_branch_protection", target, repo, {"branch": branch, "required_approvals": required_approvals})
    return r.json()


# ---------------------------------------------------------------------------
# CLI — the conductor / Facets handoff surface
# ---------------------------------------------------------------------------
def _main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="agents_core.forgejo", description="Conductor repo management (Forgejo on BRIX)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("create", help="create a repo (default org: lapis)")
    pc.add_argument("name")
    pc.add_argument("--owner", default=LAPIS_ORG, help="org/user namespace (default: lapis)")
    pc.add_argument("--desc", default="")
    pc.add_argument("--public", action="store_true", help="create public (default private)")
    pc.add_argument("--push", metavar="LOCAL_PATH", help="also push this local git tree")

    pp = sub.add_parser("push", help="create + push a local repo (default org: lapis)")
    pp.add_argument("name")
    pp.add_argument("local_path")
    pp.add_argument("--owner", default=LAPIS_ORG)
    pp.add_argument("--desc", default="")
    pp.add_argument("--public", action="store_true")

    pl = sub.add_parser("list", help="list repos under an org/user")
    pl.add_argument("--owner", default=LAPIS_ORG)

    pd = sub.add_parser("delete", help="DESTRUCTIVE: delete a repo (needs --confirm)")
    pd.add_argument("name")
    pd.add_argument("--owner", default=LAPIS_ORG)
    pd.add_argument("--confirm", action="store_true")

    pw = sub.add_parser("webhook", help="add a webhook to a repo")
    pw.add_argument("name")
    pw.add_argument("url")
    pw.add_argument("--owner", default=LAPIS_ORG)
    pw.add_argument("--secret", default=os.environ.get("FORGEJO_WEBHOOK_SECRET", ""))
    pw.add_argument("--events", default="push")

    a = p.parse_args(argv)
    if a.cmd == "create":
        if a.push:
            out = create_and_push(a.name, a.push, owner=a.owner, description=a.desc, private=not a.public)
        else:
            out = create_repo(a.name, description=a.desc, private=not a.public, owner=a.owner)
        print(json.dumps({"created": f"{a.owner}/{a.name}", "result": out.get("clone_url") or out.get("clone_url", out.get("html_url"))}, default=str))
    elif a.cmd == "push":
        out = create_and_push(a.name, a.local_path, owner=a.owner, description=a.desc, private=not a.public)
        print(json.dumps({"pushed": f"{a.owner}/{a.name}", "clone_url": out["clone_url"]}))
    elif a.cmd == "list":
        for r in list_repos(owner=a.owner):
            print(f"{r['full_name']}\t{'private' if r['private'] else 'public'}\t{r.get('description','')}")
    elif a.cmd == "delete":
        delete_repo(a.name, owner=a.owner, confirm=a.confirm)
        print(json.dumps({"deleted": f"{a.owner}/{a.name}"}))
    elif a.cmd == "webhook":
        out = create_webhook(a.name, a.url, a.secret, events=a.events.split(","), owner=a.owner)
        print(json.dumps({"webhook": out.get("id"), "repo": f"{a.owner}/{a.name}"}))


if __name__ == "__main__":
    _main()
