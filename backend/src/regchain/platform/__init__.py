"""Production-path security baseline: request identity, RBAC, audit chain, tenant storage,
secrets, log redaction and retention. Architecture only; nothing here talks to a cloud.

The single-operator pilot workspace (``regchain.pilot.workspace``) is a loopback server with
one bearer token, one tenant and one person, whose roles come from OPERATOR_ROLES rather than a
login; its evidence model already refuses the things these modules exist to refuse at scale
(path escape, run ids that are not 32 hex characters, prompts or answers in retained logs).
This package holds the pieces a multi-tenant deployment needs on top, written so the wiring is
mechanical:

* ``context``   who acts, for which tenant, under which request id (a ``ContextVar``);
* ``rbac``      roles, permissions, ``require()`` and tenant isolation checks;
* ``audit``     append-only JSONL audit log with a hash chain (``regchain.evidence.digest``);
* ``storage``   tenant-prefixed, content-addressed document store with containment checks;
* ``secrets``   allowlisted environment secrets and masking of their values in log lines;
* ``redaction`` per-level allowlists for ai-calls.jsonl entries and a free-text scrubber;
* ``retention`` retention window, export of run directories to a zip, audited deletion.

These modules do not import the workspace. Since v0.17 the workspace imports ``context``,
``rbac``, ``audit``, ``redaction`` and ``retention``: request ids on every response, audit entries
per action, ai-calls.jsonl redacted before it leaves the machine (both exports, since v0.18),
retention endpoints. ``storage`` and ``secrets`` outside the redactor are not wired yet.
"""
