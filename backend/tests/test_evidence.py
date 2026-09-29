import copy
import unittest
from regchain.evidence import canonical_bytes, digest, make_event, verify_chain

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.first = make_event({"model": "fixture-v1", "source_hash": "a"*64, "confidence": "0.93"})
        self.second = make_event({"assessment": "INSUFFICIENT_EVIDENCE"}, self.first["event_hash"])
        self.events = [self.first, self.second]

    def valid(self, events):
        return verify_chain(events, self.second["event_hash"], 2)

    def test_valid_chain(self):
        self.assertTrue(self.valid(self.events))

    def test_mutation_detected(self):
        altered = copy.deepcopy(self.events)
        altered[0]["payload"]["model"] = "different"
        self.assertFalse(self.valid(altered))

    def test_tail_deletion_detected(self):
        self.assertFalse(self.valid(self.events[:1]))

    def test_reordering_detected(self):
        self.assertFalse(self.valid(list(reversed(self.events))))

    def test_recomputed_chain_rejected_by_trusted_head(self):
        first = make_event({"forged": True})
        second = make_event(self.second["payload"], first["event_hash"])
        self.assertFalse(self.valid([first, second]))

    def test_canonical_key_order(self):
        self.assertEqual(digest({"b":2,"a":1}), digest({"a":1,"b":2}))
        self.assertEqual(canonical_bytes({"a":1}), b'{"a":1}')

    def test_floats_and_non_string_keys_rejected(self):
        for value in [float("nan"), 0.93, {1:"bad"}, {"x":(1,2)}]:
            with self.assertRaises(ValueError):
                canonical_bytes(value)

    def test_event_snapshots_caller_payload(self):
        payload = {"nested": ["original"]}
        event = make_event(payload)
        payload["nested"][0] = "changed"
        self.assertEqual(event["payload"]["nested"], ["original"])

    def test_invalid_previous_hash(self):
        with self.assertRaises(ValueError):
            make_event({}, "invalid")

if __name__ == "__main__":
    unittest.main()
