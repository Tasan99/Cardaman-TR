"""Evaluation harness (v0.17): expert-labelled cases scored against the pilot's own output.

A unit test proves the code does what its author meant; this package measures how often the
analysis is right. The two are reported apart, and a run's report names its mode: with the
rules provider only the deterministic parts (extraction rules, entity gate, retrieval order)
are exercised and every AI judgement is UNKNOWN; with Ollama the whole pipeline runs.
"""
