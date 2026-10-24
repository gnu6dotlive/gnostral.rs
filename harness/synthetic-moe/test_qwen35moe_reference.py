"""Properties of the float64 qwen35moe reference that do not depend on any engine."""

from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:  # CI installs numpy for this suite; a bare checkout without it skips loudly
    np = None


@unittest.skipUnless(np is not None, "numpy is required for the reference suite")
class ReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        import qwen35moe_reference as ref
        self.ref = ref

    def test_chat_tokens_follow_the_fixture_vocabulary(self) -> None:
        ids = self.ref.chat_tokens("ab")
        expected = [256, *b"user\nab", 257, *b"\n", 256, *b"assistant\n"]
        self.assertEqual(ids, expected)
        self.assertEqual(self.ref.chat_tokens("é")[6:8], list("é".encode()))

    def test_delta_rule_recalls_what_it_wrote(self) -> None:
        dk = dv = 4
        keys = np.eye(dk)[:, None, :]               # T=4, one head, orthonormal keys
        values = np.arange(16, dtype=float).reshape(4, 1, dv)
        zeros, ones = np.zeros((4, 1)), np.ones((4, 1))
        out = self.ref.gated_delta_rule(keys, keys, values, zeros, ones)
        # with no decay and beta = 1, reading key t right after writing it returns v_t exactly
        np.testing.assert_allclose(out[:, 0], values[:, 0], atol=1e-12)

    def test_delta_rule_decay_and_partial_write(self) -> None:
        k = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        v = np.array([[[2.0, 4.0]], [[1.0, 1.0]]])
        g = np.log(np.array([[1.0], [0.5]]))
        beta = np.array([[1.0], [0.5]])
        q = np.array([[[1.0, 0.0]], [[1.0, 0.0]]])
        out = self.ref.gated_delta_rule(q, k, v, g, beta)
        # step 2 halves the state; key 2 is orthogonal to key 1, so reading key 1 gives half of v_1
        np.testing.assert_allclose(out[1, 0], [1.0, 2.0], atol=1e-12)

    def test_delta_rule_overwrites_a_repeated_key(self) -> None:
        k = np.array([[[1.0, 0.0]], [[1.0, 0.0]]])
        v = np.array([[[3.0, 1.0]], [[5.0, -2.0]]])
        out = self.ref.gated_delta_rule(k, k, v, np.zeros((2, 1)), np.ones((2, 1)))
        # the delta rule replaces the value stored under a key (a plain linear-attention sum would give v1 + v2)
        np.testing.assert_allclose(out[1, 0], [5.0, -2.0], atol=1e-12)

    def test_tiled_order_matches_the_converter_permutation(self) -> None:
        # llama.cpp conversion/qwen.py _reorder_v_heads: reshape [k_heads, v_per_k, d], swap the first two axes
        for k_heads, v_per_k in [(2, 2), (2, 4), (4, 1), (3, 2)]:
            v_heads = k_heads * v_per_k
            grouped_key = np.arange(v_heads) // v_per_k            # HF repeat_interleave pairing
            tiled = grouped_key.reshape(k_heads, v_per_k).transpose(1, 0).reshape(v_heads)
            np.testing.assert_array_equal(tiled, self.ref.tiled_key_of_value(v_heads, k_heads))

    def test_attention_first_position_sees_only_itself(self) -> None:
        rng = np.random.default_rng(0)
        hidden, heads, hd, rd = 8, 2, 4, 2
        hp = {"attention.head_count": heads, "attention.head_count_kv": 1, "attention.key_length": hd,
              "attention.layer_norm_rms_epsilon": 1e-6, "rope.dimension_count": rd, "rope.freq_base": 1e4}
        w = {"attn_q.weight": rng.normal(size=(2 * heads * hd, hidden)), "attn_k.weight": rng.normal(size=(hd, hidden)),
             "attn_v.weight": rng.normal(size=(hd, hidden)), "attn_output.weight": rng.normal(size=(hidden, heads * hd)),
             "attn_q_norm.weight": np.ones(hd), "attn_k_norm.weight": np.ones(hd)}
        h = rng.normal(size=(3, hidden))
        out = self.ref.attention(h, w, "", hp)
        gate = (h[0] @ w["attn_q.weight"].T).reshape(heads, 2 * hd)[:, hd:].reshape(-1)
        v0 = h[0] @ w["attn_v.weight"].T
        expected = (np.tile(v0, heads) * self.ref.sigmoid(gate)) @ w["attn_output.weight"].T
        np.testing.assert_allclose(out[0], expected, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
