from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from glossary import (
    active_terms,
    add_terms,
    build_initial_prompt,
    default_glossary,
    inactive_terms,
    load_glossary,
    normalize,
    remove_term,
    save_glossary,
    set_all_active,
    set_term_active,
    split_terms,
)


class SplitTermsTests(unittest.TestCase):
    def test_splits_on_every_supported_separator(self):
        raw = "ArchiCAD、subagent,容積率，Claude Code;Notion／x\nsrt"
        self.assertEqual(
            split_terms(raw),
            ["ArchiCAD", "subagent", "容積率", "Claude Code", "Notion／x", "srt"],
        )

    def test_keeps_inner_spaces_but_collapses_runs(self):
        self.assertEqual(split_terms("  Claude   Code  "), ["Claude Code"])

    def test_drops_case_insensitive_duplicates_within_one_input(self):
        self.assertEqual(split_terms("ArchiCAD、archicad"), ["ArchiCAD"])

    def test_empty_input_yields_no_terms(self):
        self.assertEqual(split_terms(""), [])
        self.assertEqual(split_terms("、,，\n"), [])


class GlossaryMutationTests(unittest.TestCase):
    def setUp(self):
        self.data = default_glossary()

    def test_add_terms_returns_only_new_entries(self):
        self.assertEqual(add_terms(self.data, "ArchiCAD、subagent"), ["ArchiCAD", "subagent"])
        self.assertEqual(add_terms(self.data, "subagent、Notion"), ["Notion"])
        self.assertEqual(active_terms(self.data), ["ArchiCAD", "subagent", "Notion"])

    def test_re_adding_an_inactive_term_reactivates_it_without_duplicating(self):
        add_terms(self.data, "ArchiCAD")
        set_term_active(self.data, "ArchiCAD", False)
        self.assertEqual(add_terms(self.data, "archicad"), [])
        self.assertEqual(active_terms(self.data), ["ArchiCAD"])
        self.assertEqual(len(self.data["terms"]), 1)

    def test_toggling_moves_a_term_between_the_two_columns(self):
        add_terms(self.data, "ArchiCAD、subagent")
        set_term_active(self.data, "subagent", False)
        self.assertEqual(active_terms(self.data), ["ArchiCAD"])
        self.assertEqual(inactive_terms(self.data), ["subagent"])

    def test_set_term_active_matches_case_insensitively(self):
        add_terms(self.data, "ArchiCAD")
        self.assertTrue(set_term_active(self.data, "ARCHICAD", False))
        self.assertEqual(active_terms(self.data), [])

    def test_set_term_active_reports_unknown_terms(self):
        self.assertFalse(set_term_active(self.data, "missing", True))

    def test_remove_term_deletes_permanently(self):
        add_terms(self.data, "ArchiCAD、subagent")
        self.assertTrue(remove_term(self.data, "archicad"))
        self.assertFalse(remove_term(self.data, "archicad"))
        self.assertEqual(self.data["terms"], [{"text": "subagent", "active": True}])

    def test_set_all_active_covers_both_directions(self):
        add_terms(self.data, "a、b、c")
        set_all_active(self.data, False)
        self.assertEqual(active_terms(self.data), [])
        set_all_active(self.data, True)
        self.assertEqual(active_terms(self.data), ["a", "b", "c"])


class InitialPromptTests(unittest.TestCase):
    def test_prompt_joins_prefix_then_active_terms(self):
        data = default_glossary()
        data["prefix"] = "中文為主。"
        add_terms(data, "ArchiCAD、subagent")
        set_term_active(data, "subagent", False)
        self.assertEqual(build_initial_prompt(data), "中文為主。 ArchiCAD")

    def test_terms_come_last_so_left_truncation_keeps_them(self):
        data = default_glossary()
        data["prefix"] = "前綴"
        add_terms(data, "尾詞")
        self.assertTrue(build_initial_prompt(data).endswith("尾詞"))

    def test_prompt_is_empty_when_nothing_is_configured(self):
        data = default_glossary()
        data["prefix"] = ""
        self.assertEqual(build_initial_prompt(data), "")

    def test_prefix_alone_is_a_valid_prompt(self):
        data = default_glossary()
        data["prefix"] = "只有前綴"
        self.assertEqual(build_initial_prompt(data), "只有前綴")


class NormalizeTests(unittest.TestCase):
    def test_accepts_a_plain_list_of_strings(self):
        data = normalize({"terms": ["ArchiCAD", "subagent"]})
        self.assertEqual(active_terms(data), ["ArchiCAD", "subagent"])

    def test_drops_blank_and_duplicate_entries(self):
        data = normalize({"terms": [{"text": " "}, {"text": "a"}, {"text": "A"}, 42]})
        self.assertEqual([entry["text"] for entry in data["terms"]], ["a"])

    def test_unusable_input_falls_back_to_defaults(self):
        self.assertEqual(normalize("oops"), default_glossary())
        self.assertEqual(normalize(None), default_glossary())

    def test_accepts_a_hand_written_bare_array(self):
        data = normalize([{"text": "ArchiCAD"}, "軸網"])
        self.assertEqual(active_terms(data), ["ArchiCAD", "軸網"])

    def test_accepts_enabled_as_a_synonym_for_active(self):
        data = normalize([{"text": "a", "enabled": False}, {"text": "b", "enabled": True}])
        self.assertEqual(active_terms(data), ["b"])
        self.assertEqual(inactive_terms(data), ["a"])

    def test_preserves_the_active_flag(self):
        data = normalize({"terms": [{"text": "a", "active": False}]})
        self.assertEqual(inactive_terms(data), ["a"])


class GlossaryStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "glossary.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_round_trip_survives_a_restart(self):
        data = default_glossary()
        add_terms(data, "ArchiCAD、subagent")
        set_term_active(data, "subagent", False)
        data["prefix"] = "自訂前綴"
        save_glossary(self.path, data)

        reloaded = load_glossary(self.path)
        self.assertEqual(reloaded["prefix"], "自訂前綴")
        self.assertEqual(active_terms(reloaded), ["ArchiCAD"])
        self.assertEqual(inactive_terms(reloaded), ["subagent"])

    def test_missing_file_yields_defaults(self):
        self.assertEqual(load_glossary(self.path), default_glossary())

    def test_corrupt_file_does_not_raise(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(load_glossary(self.path), default_glossary())

    def test_saved_file_is_readable_utf8_json(self):
        data = default_glossary()
        add_terms(data, "容積率")
        save_glossary(self.path, data)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(payload["terms"], [{"text": "容積率", "active": True}])

    def test_save_leaves_no_temporary_files_behind(self):
        save_glossary(self.path, default_glossary())
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["glossary.json"])

    def test_a_shrinking_save_keeps_a_backup(self):
        data = default_glossary()
        add_terms(data, "a、b、c、d、e")
        save_glossary(self.path, data)
        backup = save_glossary(self.path, default_glossary())
        self.assertIsNotNone(backup)
        self.assertEqual(active_terms(load_glossary(backup)), ["a", "b", "c", "d", "e"])

    def test_deleting_a_single_term_does_not_spam_backups(self):
        data = default_glossary()
        add_terms(data, "a、b、c")
        save_glossary(self.path, data)
        remove_term(data, "a")
        self.assertIsNone(save_glossary(self.path, data))


if __name__ == "__main__":
    unittest.main()
