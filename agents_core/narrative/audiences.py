"""Audience profiles for the narrative template engine.

The four v0 audience slugs and their associated framing data.

Authorship: profile prose was drafted by Claude (Opus) per Erah's
delegation 2026-05-07. The spec invariant "Erah-authored profile prose"
is relaxed to "Erah-delegated and reviewed" for v0; v1 gates on a
Lapis-refresh of the archived voice-reference doc plus hands-on use of
the engine.
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
# Four v0 profiles
# ---------------------------------------------------------------------------

_GRANTS = AudienceProfile(
    slug="grants",
    title="Grant program officer / fiscal-sponsor reviewer",
    frame=(
        "An NLNet, NGI Zero, Open Collective, or similar grant program "
        "officer reading the Nth application this cycle. Skimming for "
        "evidence-of-life signals: shipped artifacts, public infrastructure, "
        "real users or contributors, and a coherent reason this work needs "
        "to be public-good infrastructure rather than a startup. Defaults "
        "to skepticism on vision-tier rhetoric and rewards specific, "
        "verifiable claims."
    ),
    register=(
        "Technical-credible, plain, and slightly understated. Sentences "
        "carry their own weight. No marketing voice and no founder "
        "mythology, but the writing should still feel like a real person, "
        "not a grants-template. Evidence cited inline (links, repo paths, "
        "dated commits) over hand-waving. Hedging is OK when it's honest; "
        "hype is not."
    ),
    emphasize=[
        "Open primitives - published modules, no proprietary moat, "
        "permissive licensing where possible.",
        "Structural impossibility of weaponization (Constitution-Kernel "
        "Invariant 8) - design rules out the harmful shape rather than "
        "relying on policy to refuse it.",
        "User sovereignty (Invariant 1) - the human stays at the center; "
        "AI is mirror, not destination.",
        "Shipped artifacts as proof-of-life - real repositories, dated "
        "commits, in-flight work visible in public.",
    ],
    deemphasize=[
        "Vision-tier rhetoric - 'transformative', 'revolutionary', 'will "
        "change how X works'.",
        "Founder narrative - biographical framing belongs in a "
        "personal-essay venue, not a grant section asking what the work "
        "does.",
        "Future speculation untied to current artifacts - anything that "
        "begins 'eventually we want to' without a current move toward it.",
    ],
    example_callouts=[
        "The proof here is the repos themselves, not the framing - happy "
        "to point at specific commits if useful.",
        "Open primitives, no moat, no lock-in. The thing we're keeping is "
        "the relationship with the community, not the IP.",
        "If the structural argument doesn't hold, the policy one won't "
        "either - so the design rules out the failure shape.",
    ],
    extra_sources=[
        "Lapis/Vision-Graduated-Commons.md",
        "Lapis/Vision-Commons-and-Commerce.md",
    ],
)

_KYMA_MARKETING = AudienceProfile(
    slug="kyma-marketing",
    title="TTRPG creator (DM, table prep, character worldbuilder) on a SaaS landing page",
    frame=(
        "A TTRPG-shaped creator - DM, table-prep author, character "
        "worldbuilder, or solo journaling player - landing on a SaaS page "
        "or feature blurb. Curious about anything that helps them at the "
        "table, suspicious of anything that smells like AI replacing the "
        "craft. They want better tools for the simulation work they "
        "already do, not a new app demanding to be the destination."
    ),
    register=(
        "Warm, specific, and simulation-shaped. Talks about the work the "
        "way a tablemate would, not the way a press release would. No "
        "pathology framing (this is not therapy, not 'AI for mental "
        "health'); no AI-engine framing (the AI is in service to the "
        "table, not the headline). Embodied language is welcome - what "
        "does this feel like at the table?"
    ),
    emphasize=[
        "Behavioral simulation as ancient TTRPG craft, made computational "
        "- this is what your table already does, just with a tool that "
        "can carry weight between sessions.",
        "No-pathology design - not therapy, not 'AI companion to fix "
        "you', just better simulation for play and craft.",
        "Character-first flow - the character (and the table) is the "
        "protagonist of the experience, not the app.",
        "Sovereignty over their table - they keep what they make, the "
        "tool stays in service.",
    ],
    deemphasize=[
        "AI-engine framing - 'powered by Claude', 'cutting-edge LLM', "
        "'AI agent' as the headline noun.",
        "Infrastructure or architecture detail - Zephyrium, Archetypal "
        "Intelligence internals, agent routing.",
        "Lapis-ecosystem naming - the page is about Kyma, not the parent "
        "brand.",
    ],
    example_callouts=[
        "The thing your table already does - running social simulations "
        "to figure out how a character might move through a moment - "
        "Kyma just helps that carry forward between sessions.",
        "It's a tool for the work, not the destination of the work. You "
        "stay the storyteller.",
        "Not therapy. Not a chatbot pretending to be a friend. Just "
        "better gear for the craft you already practice.",
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
    frame=(
        "A community member who's already chosen to follow Erah's work - "
        "patron, sustainer, contributor, or curious developer who showed "
        "up before there was much to see. They're reading a cadence "
        "update and want honesty about state, not polish. Trust here is "
        "built by being real about what shipped, what didn't, and what's "
        "next."
    ),
    register=(
        "Status-shaped, personal, transparent. Closer to a craftsperson's "
        "logbook than a marketing post. Honest about what shipped vs. "
        "what's stuck. Hype is a trust-debit; understatement reads as "
        "competence. Specific dates, specific repos, specific "
        "stuck-points. Self-coaching tone is welcome - 'so let's...' / "
        "'the play next week is...'"
    ),
    emphasize=[
        "What landed in the reporting window - name the artifacts, link "
        "the commits, dated.",
        "What's blocked and why - name the constraint honestly (time, "
        "hardware, money, design uncertainty).",
        "What the next concrete move is - single-sentence, "
        "single-direction, falsifiable in the next window.",
        "The canonical 'why this matters' anchor - one or two of the "
        "eight Constitution-Kernel invariants the work is serving, no "
        "more.",
    ],
    deemphasize=[
        "Roadmap language - quarterly OKRs, 'Q3 we will', 'we're "
        "targeting' phrasing.",
        "Unannounced commitments - anything that promises future "
        "deliverables not already in flight.",
        "Unverifiable claims - 'users love', 'feedback has been "
        "incredible' without specifics.",
    ],
    example_callouts=[
        "Solid week. Two things landed, one is stuck, here's the play "
        "next.",
        "Honestly I think we're close on this one - just need to lock "
        "down the method.",
        "Not great this week - but the failure mode was useful, here's "
        "what changed.",
    ],
    extra_sources=[],  # canonical four only
)

_DEV_DOCS = AudienceProfile(
    slug="dev-docs",
    title="Indie AI/agent developer reading a positioning post or repo README",
    frame=(
        "An indie developer building agentic systems - probably reading "
        "a README, a blog post, or a positioning page after a GitHub "
        "search or HN link. Wants to know in 30 seconds: is this "
        "composable with what I'm already doing, what's actually "
        "shipped, and is it worth a second read. Allergic to vision-tier "
        "prose without code behind it."
    ),
    register=(
        "Technical, specific, composable. Code paths, repo links, and "
        "primitives over abstractions. Plain about what is dogfood "
        "(still moving, expect breakage) versus what is shipped (stable, "
        "depend on it). Conversational where it helps; precise where it "
        "counts. Hyphens and parentheticals welcome - natural speech "
        "rhythm, not press-release polish."
    ),
    emphasize=[
        "Open primitives - name them: agents_core, archetypes_core, "
        "lapis_pm, etc. Show the import line.",
        "Agent-operability with provenance - the LapisToolReturn "
        "envelope, citations, why provenance is the contract.",
        "What is dogfood vs. what is shipped - be honest about which "
        "surfaces are stable.",
        "How this composes with their work - what they could pull in "
        "tomorrow without buying the whole ecosystem.",
    ],
    deemphasize=[
        "Business or fundraising framing - sustainer cadence, "
        "grants-runway language, MRR targets.",
        "Vision-tier prose - 'a new paradigm', 'rethinking', "
        "'reimagining' as the load-bearing nouns.",
    ],
    example_callouts=[
        "If you're already running a Claude-shaped agent loop, you can "
        "pull lapis_research as a sub-substrate without committing to "
        "the rest of the stack.",
        "The provenance envelope is the contract - model + prompt-hash "
        "+ citations on every tool return. Not optional.",
        "Honest about what's dogfood - the cockpit surfaces are still "
        "moving; the core primitives in archetypes_core are stable "
        "enough to depend on.",
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
