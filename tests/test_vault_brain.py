"""Unit tests for the pure parts of vault_brain (no Ollama needed).

Run: python3 -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vault_brain as vb  # noqa: E402


class VectorTests(unittest.TestCase):
    def test_normalize_unit_length(self):
        v = vb.normalize([3.0, 4.0])
        self.assertAlmostEqual(v[0], 0.6)
        self.assertAlmostEqual(v[1], 0.8)

    def test_normalize_zero_vector_is_safe(self):
        self.assertEqual(vb.normalize([0.0, 0.0]), [0.0, 0.0])

    def test_dot(self):
        self.assertEqual(vb.dot([1, 2, 3], [4, 5, 6]), 32)

    def test_blob_round_trip(self):
        v = [0.25, -1.5, 3.0]
        self.assertEqual(list(vb.from_blob(vb.to_blob(v))), v)

    def test_mean_vector(self):
        self.assertIsNone(vb.mean_vector([]))
        m = vb.mean_vector([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(vb.dot(m, m), 1.0)
        self.assertAlmostEqual(m[0], m[1])


class FrontmatterTests(unittest.TestCase):
    def test_parses_keys_and_body(self):
        text = "---\ntype: principle\nstatus: seed  # comment\ntitle: 'A: B'\n---\n\n# Hello\nbody"
        meta, body = vb.parse_frontmatter(text)
        self.assertEqual(meta["type"], "principle")
        self.assertEqual(meta["status"], "seed")
        self.assertEqual(meta["title"], "A: B")
        self.assertTrue(body.startswith("# Hello"))

    def test_no_frontmatter(self):
        meta, body = vb.parse_frontmatter("just text")
        self.assertEqual(meta, {})
        self.assertEqual(body, "just text")

    def test_unclosed_frontmatter(self):
        meta, body = vb.parse_frontmatter("---\ntype: x\nno end")
        self.assertEqual(meta, {})


class AreaTests(unittest.TestCase):
    def test_areas(self):
        self.assertEqual(vb.area_of("Note.md"), "(root)")
        self.assertEqual(vb.area_of("Cars/A4.md"), "Cars")
        self.assertEqual(vb.area_of("Forge Labs/Mind Lab/Principles/X.md"), "Forge Labs/Mind Lab")
        self.assertEqual(vb.area_of("Forge Labs/X.md"), "Forge Labs")


class ChunkTests(unittest.TestCase):
    def test_short_text_is_one_piece(self):
        self.assertEqual(vb.split_long("short"), ["short"])

    def test_long_text_respects_limit(self):
        text = "\n\n".join("word " * 60 for _ in range(20))
        pieces = vb.split_long(text, limit=500, overlap=50)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(p) <= 500 for p in pieces))

    def test_giant_paragraph_hard_split(self):
        pieces = vb.split_long("x" * 1200, limit=500, overlap=100)
        self.assertTrue(all(len(p) <= 500 for p in pieces))
        self.assertGreaterEqual(sum(len(p) for p in pieces), 1200)

    def test_chunk_by_headings_skips_sources_and_title(self):
        body = (
            "# My Note\nintro text\n\n## Idea\nidea text\n### Detail\ndetail text\n"
            "## Sources\n- https://example.com\n"
        )
        chunks = vb.chunk_note("My Note", body)
        names = [name for _, name, _ in chunks]
        self.assertIn("Idea", names)
        self.assertIn("Detail", names)
        self.assertNotIn("Sources", names)
        paths = {name: path for path, name, _ in chunks}
        self.assertEqual(paths["Detail"], "Idea > Detail")

    def test_headings_inside_code_fence_ignored(self):
        body = "## Code\n```\n# not a heading\n```\n"
        chunks = vb.chunk_note("T", body)
        self.assertEqual(len(chunks), 1)
        self.assertIn("# not a heading", chunks[0][2])


class TextTests(unittest.TestCase):
    def test_keywords_drop_short_and_stopwords(self):
        kw = vb.keywords("The quick brown fox is on it")
        self.assertIn("quick", kw)
        self.assertIn("brown", kw)
        self.assertNotIn("the", kw)
        self.assertNotIn("is", kw)

    def test_check_citations(self):
        chunks = [{"title": "Loss Aversion"}, {"title": "Kelly Criterion"}]
        answer = "See [[Loss Aversion]] and [[Kelly Criterion|Kelly]] and [[Made Up]] and [[loss aversion#Why]]."
        cited, bad = vb.check_citations(answer, chunks)
        self.assertEqual(cited, ["Loss Aversion", "Kelly Criterion", "Made Up", "loss aversion"])
        self.assertEqual(bad, ["Made Up"])


class IndexSmokeTests(unittest.TestCase):
    def test_iter_notes_on_temp_vault(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "Lab"))
            os.makedirs(os.path.join(d, ".git"))
            for rel, text in (("Lab/A.md", "# A\ntext"), (".git/B.md", "skip me")):
                with open(os.path.join(d, rel), "w") as f:
                    f.write(text)
            old = vb.VAULT
            vb.VAULT = d
            try:
                found = [rel for rel, _ in vb.iter_notes()]
            finally:
                vb.VAULT = old
            self.assertEqual(found, ["Lab/A.md"])


if __name__ == "__main__":
    unittest.main()
