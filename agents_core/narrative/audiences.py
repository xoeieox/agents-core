"""Audience profiles for the narrative template engine.

The four v0 audience slugs and their associated framing data.
Profile prose (frame, register, emphasize, deemphasize, example_callouts)
is Erah-authored. Fixer ships TODO placeholders; Erah fills them in on the
same PR branch before merge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudienceProfile:
    slug: str
    title: str
    frame: str
    register: str
    emphasize: list[str]
    deemphasize: list[str]
    example_callouts: list[str]
    extra_sources: list[str]  # vault-relative paths under /srv/git/inertia-vault-working/


# ---------------------------------------------------------------------------
# Four v0 profiles — prose marked TODO(Erah) pending Erah-authored bodies
# ---------------------------------------------------------------------------

_GRANTS = AudienceProfile(
    slug="grants",
    title="Grant program officer / fiscal-sponsor reviewer",
    # TODO(Erah): author grants profile body — placeholder will not produce useful drafts
    frame=(
        "TODO(Erah): frame for grants audience — "
        "NLNet/Open Collective program officer reading hundreds of applications."
    ),
    register=(
        "TODO(Erah): register for grants audience — "
        "technical-credible, no marketing voice, evidence-of-life signals."
    ),
    emphasize=[
        "TODO(Erah): emphasize[0] — open primitives, no proprietary moat",
        "TODO(Erah): emphasize[1] — structural impossibility of weaponization (Invariant 8)",
        "TODO(Erah): emphasize[2] — user sovereignty (Invariant 1)",
        "TODO(Erah): emphasize[3] — shipped artifacts as proof-of-life",
    ],
    deemphasize=[
        "TODO(Erah): deemphasize[0] — vision-tier rhetoric",
        "TODO(Erah): deemphasize[1] — founder narrative",
        "TODO(Erah): deemphasize[2] — future speculation",
    ],
    example_callouts=[
        "TODO(Erah): example_callouts[0] — characteristic grants phrasing",
    ],
    extra_sources=[
        "Lapis/Vision-Graduated-Commons.md",
        "Lapis/Vision-Commons-and-Commerce.md",
    ],
)

_KYMA_MARKETING = AudienceProfile(
    slug="kyma-marketing",
    title="TTRPG creator (DM, table prep, character worldbuilder) on a SaaS landing page",
    # TODO(Erah): author kyma-marketing profile body — placeholder will not produce useful drafts
    frame=(
        "TODO(Erah): frame for kyma-marketing audience — "
        "TTRPG creator seeing a SaaS landing-page or feature blurb."
    ),
    register=(
        "TODO(Erah): register for kyma-marketing audience — "
        "warm, specific, simulation-shaped, no pathology framing."
    ),
    emphasize=[
        "TODO(Erah): emphasize[0] — behavioral simulation as ancient TTRPG craft made computational",
        "TODO(Erah): emphasize[1] — no-pathology design",
        "TODO(Erah): emphasize[2] — character-first flow",
        "TODO(Erah): emphasize[3] — sovereignty over their table",
    ],
    deemphasize=[
        "TODO(Erah): deemphasize[0] — AI-engine framing",
        "TODO(Erah): deemphasize[1] — infrastructure / architecture detail",
        "TODO(Erah): deemphasize[2] — Lapis-ecosystem naming",
    ],
    example_callouts=[
        "TODO(Erah): example_callouts[0] — characteristic kyma-marketing phrasing",
    ],
    extra_sources=[
        "Lapis/Products/Kyma/MOC-Kyma.md",
        "Lapis/Products/Kyma/Documentation/MVP-Scope.md",
        "Lapis/Products/Kyma/Documentation/Full-Loop-Vision.md",
    ],
)

_SUSTAINER_UPDATE = AudienceProfile(
    slug="sustainer-update",
    title="Community member building around Lapis who reads a cadence post from Erah",
    # TODO(Erah): author sustainer-update profile body — placeholder will not produce useful drafts
    frame=(
        "TODO(Erah): frame for sustainer-update audience — "
        "someone in the community building around Lapis who reads a cadence post from Erah."
    ),
    register=(
        "TODO(Erah): register for sustainer-update audience — "
        "status-shaped, personal, honest about what shipped vs. what's stuck, no hype."
    ),
    emphasize=[
        "TODO(Erah): emphasize[0] — what landed in the reporting window",
        "TODO(Erah): emphasize[1] — what's blocked and why",
        "TODO(Erah): emphasize[2] — what the next concrete move is",
        "TODO(Erah): emphasize[3] — canonical 'why this matters' anchor (one or two invariants)",
    ],
    deemphasize=[
        "TODO(Erah): deemphasize[0] — roadmap language",
        "TODO(Erah): deemphasize[1] — unannounced commitments",
        "TODO(Erah): deemphasize[2] — unverifiable claims",
    ],
    example_callouts=[
        "TODO(Erah): example_callouts[0] — characteristic sustainer-update phrasing",
    ],
    extra_sources=[],  # canonical four only
)

_DEV_DOCS = AudienceProfile(
    slug="dev-docs",
    title="Indie AI/agent developer reading a positioning post or repo README",
    # TODO(Erah): author dev-docs profile body — placeholder will not produce useful drafts
    frame=(
        "TODO(Erah): frame for dev-docs audience — "
        "indie AI/agent developer reading a positioning post or repo README."
    ),
    register=(
        "TODO(Erah): register for dev-docs audience — "
        "technical, specific, composable."
    ),
    emphasize=[
        "TODO(Erah): emphasize[0] — open primitives",
        "TODO(Erah): emphasize[1] — agent-operability with provenance",
        "TODO(Erah): emphasize[2] — what is dogfood vs. what is shipped",
        "TODO(Erah): emphasize[3] — how this composes with their work",
    ],
    deemphasize=[
        "TODO(Erah): deemphasize[0] — business / fundraising framing",
        "TODO(Erah): deemphasize[1] — vision-tier prose",
    ],
    example_callouts=[
        "TODO(Erah): example_callouts[0] — characteristic dev-docs phrasing",
    ],
    extra_sources=[
        "Lapis/State-of-Lapis-2026-04-25.md",
        "Lapis/Build-Stack-Roadmap.md",
    ],
)

AUDIENCE_REGISTRY: dict[str, AudienceProfile] = {
    p.slug: p for p in [_GRANTS, _KYMA_MARKETING, _SUSTAINER_UPDATE, _DEV_DOCS]
}

VALID_SLUGS: list[str] = list(AUDIENCE_REGISTRY.keys())
