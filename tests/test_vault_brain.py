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


class AskJsonTests(unittest.TestCase):
    def test_ask_json_output(self):
        import argparse
        import contextlib
        import io
        import json
        chunks = [
            {"title": "Loss Aversion", "heading": "Why", "path": "Mind/Loss Aversion.md", "score": 0.81234, "cos": 0.7, "kw": 0.5, "text": "Note: Loss Aversion > Why\nLosses weigh about twice as much as gains."},
            {"title": "Loss Aversion", "heading": "Uses", "path": "Mind/Loss Aversion.md", "score": 0.6, "cos": 0.6, "kw": 0.0, "text": "Note: Loss Aversion > Uses\nPricing and framing."},
            {"title": "Kelly Criterion", "heading": "", "path": "Quant/Kelly Criterion.md", "score": 0.5, "cos": 0.5, "kw": 0.0, "text": "Note: Kelly Criterion\nBet sizing."},
        ]
        fakes = {
            "connect": lambda: None,
            "retrieve": lambda db, q, k=8: chunks,
            "chat": lambda messages, think=False: ("Losses hurt more [[Loss Aversion]], see [[Made Up]].",
                                                   {"eval_count": 20, "eval_duration": 2e9}),
        }
        saved = {name: getattr(vb, name) for name in fakes}
        for name, fn in fakes.items():
            setattr(vb, name, fn)
        out = io.StringIO()
        try:
            args = argparse.Namespace(question="why do losses hurt?", k=8, think=False, show_chunks=True, json=True)
            with contextlib.redirect_stdout(out):
                self.assertEqual(vb.cmd_ask(args), 0)
        finally:
            for name, fn in saved.items():
                setattr(vb, name, fn)
        data = json.loads(out.getvalue())  # the whole output is one JSON object, even with --show-chunks
        self.assertEqual(data["question"], "why do losses hurt?")
        self.assertEqual(data["cited"], ["Loss Aversion", "Made Up"])
        self.assertEqual(data["cited_not_retrieved"], ["Made Up"])
        self.assertEqual([s["title"] for s in data["sources"]], ["Loss Aversion", "Kelly Criterion"])
        self.assertEqual([s["cited"] for s in data["sources"]], [True, False])
        self.assertEqual(len(data["chunks"]), 3)
        self.assertEqual(data["timing"]["tokens"], 20)
        self.assertEqual(data["timing"]["tokens_per_s"], 10.0)

class ProgressTests(unittest.TestCase):
    def test_progress_line(self):
        self.assertEqual(vb.progress_line(0, 4, width=8), "[--------] 0/4 (  0%)")
        self.assertEqual(vb.progress_line(1, 4, width=8), "[##------] 1/4 ( 25%)")
        self.assertEqual(vb.progress_line(4, 4, "done", width=8), "[########] 4/4 (100%) done")

    def test_progress_line_edges(self):
        self.assertEqual(vb.progress_line(0, 0, width=4), "[####] 0/0 (100%)")  # empty vault reads as complete
        self.assertEqual(vb.progress_line(9, 4, width=4), "[####] 9/4 (100%)")  # never overflows the bar
        self.assertEqual(vb.progress_line(15, 100, width=10), "[#---------]  15/100 ( 15%)")

    def test_progress_is_silent_when_piped(self):
        import io
        out = io.StringIO()  # not a terminal, like a log file or CI
        p = vb.Progress(3, stream=out)
        p.update(1, "a.md")
        p.log("  embedded a.md")
        p.finish()
        self.assertEqual(out.getvalue(), "  embedded a.md\n")


if __name__ == "__main__":
    unittest.main()
