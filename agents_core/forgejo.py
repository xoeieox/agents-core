#!/usr/bin/env python3
"""Forgejo API helper — shared by all agent scripts that interact with Forgejo."""

import os
import httpx

FORGEJO_URL = os.environ.get("FORGEJO_URL", "http://203.0.113.12:3000")
FORGEJO_TOKEN = os.environ.get("FORGEJO_TOKEN", "")
OWNER = "Erah"
API = f"{FORGEJO_URL}/api/v1"
TIMEOUT = 15


def _headers():
    return {"Authorization": f"token {FORGEJO_TOKEN}", "Accept": "application/json"}


def create_repo(repo: str, description: str = "", private: bool = True) -> dict:
    """Create a new repository under the owner."""
    r = httpx.post(
        f"{API}/user/repos",
        headers=_headers(),
        json={"name": repo, "description": description, "private": private, "auto_init": False},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def add_collaborator(repo: str, username: str, permission: str = "write") -> None:
    """Add a collaborator to a repository."""
    r = httpx.put(
        f"{API}/repos/{OWNER}/{repo}/collaborators/{username}",
        headers=_headers(),
        json={"permission": permission},
        timeout=TIMEOUT,
    )
    r.raise_for_status()


def create_webhook(repo: str, target_url: str, secret: str, events: list[str] | None = None) -> dict:
    """Create a webhook on a repository."""
    r = httpx.post(
        f"{API}/repos/{OWNER}/{repo}/hooks",
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
    return r.json()


def create_pr(repo: str, title: str, head: str, base: str = "main", body: str = "") -> dict:
    """Create a pull request."""
    r = httpx.post(
        f"{API}/repos/{OWNER}/{repo}/pulls",
        headers=_headers(),
        json={"title": title, "head": head, "base": base, "body": body},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def merge_pr(repo: str, pr_number: int, method: str = "merge") -> dict:
    """Merge a pull request."""
    r = httpx.post(
        f"{API}/repos/{OWNER}/{repo}/pulls/{pr_number}/merge",
        headers=_headers(),
        json={"Do": method, "delete_branch_after_merge": True},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    if r.content:
        return r.json()
    return {"status": "ok"}


def add_comment(repo: str, issue_or_pr: int, body: str) -> dict:
    """Add a comment to an issue or PR."""
    r = httpx.post(
        f"{API}/repos/{OWNER}/{repo}/issues/{issue_or_pr}/comments",
        headers=_headers(),
        json={"body": body},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def get_open_prs(repo: str) -> list[dict]:
    """List open pull requests."""
    r = httpx.get(
        f"{API}/repos/{OWNER}/{repo}/pulls",
        headers=_headers(),
        params={"state": "open", "limit": 50},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def get_pr_diff(repo: str, pr_number: int) -> str:
    """Get the diff of a pull request."""
    r = httpx.get(
        f"{API}/repos/{OWNER}/{repo}/pulls/{pr_number}.diff",
        headers=_headers(),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.text


def create_issue(repo: str, title: str, body: str, labels: list[str] | None = None) -> dict:
    """Create an issue on a repository."""
    payload = {"title": title, "body": body}
    if labels:
        # Forgejo expects label IDs, not names — look up or skip
        pass
    r = httpx.post(
        f"{API}/repos/{OWNER}/{repo}/issues",
        headers=_headers(),
        json=payload,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def get_pr(repo: str, pr_number: int) -> dict:
    """Fetch a single pull request (includes mergeable, state, html_url)."""
    r = httpx.get(
        f"{API}/repos/{OWNER}/{repo}/pulls/{pr_number}",
        headers=_headers(),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def close_pr(repo: str, pr_number: int) -> dict:
    """Close a pull request without merging."""
    r = httpx.patch(
        f"{API}/repos/{OWNER}/{repo}/pulls/{pr_number}",
        headers=_headers(),
        json={"state": "closed"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def get_pr_comments(repo: str, pr_number: int) -> list[dict]:
    """Fetch comments on a PR (uses issues endpoint, which covers PR comments)."""
    r = httpx.get(
        f"{API}/repos/{OWNER}/{repo}/issues/{pr_number}/comments",
        headers=_headers(),
        params={"limit": 50},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def list_webhooks(repo: str) -> list[dict]:
    """List webhooks configured on a repository."""
    r = httpx.get(
        f"{API}/repos/{OWNER}/{repo}/hooks",
        headers=_headers(),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def update_webhook(repo: str, hook_id: int, events: list[str]) -> dict:
    """Update a webhook's event list."""
    r = httpx.patch(
        f"{API}/repos/{OWNER}/{repo}/hooks/{hook_id}",
        headers=_headers(),
        json={"events": events},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def set_branch_protection(repo: str, branch: str = "main", required_approvals: int = 0) -> dict:
    """Set branch protection rules."""
    r = httpx.post(
        f"{API}/repos/{OWNER}/{repo}/branch_protections",
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
    return r.json()
