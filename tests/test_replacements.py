from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from replacements import (
    active_rules,
    add_rule,
    apply_to_text,
    default_rules,
    find_rule,
    load_rules,
    normalize,
    remove_rule,
    save_rules,
    set_all_active,
    set_rule_active,
)
from transcript_utils import apply_replacement_rules


def make_record(texts):
    segments = [
        {"start": i * 2.0, "end": i * 2.0 + 2.0, "text": t} for i, t in enumerate(texts)
    ]
    return {
        "result": {"text": "".join(texts), "segments": segments},
        "original_texts": list(texts),
        "edited": False,
    }


class RuleMutationTests(unittest.TestCase):
    def setUp(self):
        self.data = default_rules()

    def test_add_rule_reports_new_entries(self):
        self.assertTrue(add_rule(self.data, "agent teams", "Agent Teams"))
        self.assertEqual(active_rules(self.data), [("agent teams", "Agent Teams")])

    def test_re_adding_the_same_search_updates_the_replacement(self):
        add_rule(self.data, "subagent", "SubAgent")
        self.assertFalse(add_rule(self.data, "subagent", "Sub-Agent"))
        self.assertEqual(active_rules(self.data), [("subagent", "Sub-Agent")])
        self.assertEqual(len(self.data["rules"]), 1)

    def test_search_text_is_case_sensitive_so_variants_coexist(self):
        add_rule(self.data, "Agent", "代理")
        add_rule(self.data, "agent", "代理人")
        self.assertEqual(len(self.data["rules"]), 2)

    def test_blank_search_is_rejected(self):
        self.assertFalse(add_rule(self.data, "   ", "x"))
        self.assertEqual(self.data["rules"], [])

    def test_empty_replacement_is_allowed_for_deletions(self):
        add_rule(self.data, "嗯嗯", "")
        self.assertEqual(active_rules(self.data), [("嗯嗯", "")])

    def test_remove_rule_deletes_permanently(self):
        add_rule(self.data, "a", "b")
        self.assertTrue(remove_rule(self.data, "a"))
        self.assertFalse(remove_rule(self.data, "a"))

    def test_toggling_excludes_a_rule_without_deleting_it(self):
        add_rule(self.data, "a", "b")
        set_rule_active(self.data, "a", False)
        self.assertEqual(active_rules(self.data), [])
        self.assertIsNotNone(find_rule(self.data, "a"))

    def test_set_all_active_covers_both_directions(self):
        add_rule(self.data, "a", "1")
        add_rule(self.data, "b", "2")
        set_all_active(self.data, False)
        self.assertEqual(active_rules(self.data), [])
        set_all_active(self.data, True)
        self.assertEqual(len(active_rules(self.data)), 2)


class ApplyToTextTests(unittest.TestCase):
    def test_rules_apply_in_order(self):
        self.assertEqual(apply_to_text("aaa", [("a", "b"), ("b", "c")]), "ccc")

    def test_no_rules_leaves_text_untouched(self):
        self.assertEqual(apply_to_text("原文", []), "原文")


class ApplyToRecordTests(unittest.TestCase):
    def test_applies_every_rule_across_every_segment(self):
        record = make_record(["用 agent teams", "跑 subagent", "無關"])
        changed, hits = apply_replacement_rules(
            record, [("agent teams", "Agent Teams"), ("subagent", "SubAgent")]
        )
        self.assertEqual(changed, 2)
        self.assertEqual(hits, {"agent teams": 1, "subagent": 1})
        self.assertEqual(record["result"]["segments"][0]["text"], "用 Agent Teams")
        self.assertEqual(record["result"]["segments"][1]["text"], "跑 SubAgent")

    def test_counts_multiple_hits_within_one_segment(self):
        record = make_record(["嗯嗯 然後 嗯嗯"])
        _, hits = apply_replacement_rules(record, [("嗯嗯", "")])
        self.assertEqual(hits["嗯嗯"], 2)
        self.assertEqual(record["result"]["segments"][0]["text"], "然後")

    def test_marks_the_record_edited_and_rebuilds_full_text(self):
        record = make_record(["甲", "乙"])
        apply_replacement_rules(record, [("甲", "丙")])
        self.assertTrue(record["edited"])
        self.assertEqual(record["result"]["text"], "丙乙")

    def test_rules_that_match_nothing_leave_the_record_untouched(self):
        record = make_record(["原文"])
        changed, hits = apply_replacement_rules(record, [("不存在", "x")])
        self.assertEqual(changed, 0)
        self.assertEqual(hits, {"不存在": 0})
        self.assertFalse(record["edited"])

    def test_empty_rule_list_is_a_no_op(self):
        record = make_record(["原文"])
        self.assertEqual(apply_replacement_rules(record, []), (0, {}))
        self.assertFalse(record["edited"])

    def test_preserves_whisper_leading_space(self):
        record = make_record([" 用 agent teams"])
        apply_replacement_rules(record, [("agent teams", "Agent Teams")])
        self.assertEqual(record["result"]["segments"][0]["text"], " 用 Agent Teams")


class NormalizeTests(unittest.TestCase):
    def test_drops_blank_and_duplicate_searches(self):
        data = normalize({"rules": [{"find": " "}, {"find": "a"}, {"find": "a"}, 7]})
        self.assertEqual([r["find"] for r in data["rules"]], ["a"])

    def test_unusable_input_falls_back_to_defaults(self):
        self.assertEqual(normalize("oops"), default_rules())
        self.assertEqual(normalize(None), default_rules())

    def test_defaults_missing_fields(self):
        data = normalize({"rules": [{"find": "a"}]})
        self.assertEqual(data["rules"], [{"find": "a", "replace": "", "active": True}])

    def test_accepts_a_hand_written_bare_array(self):
        data = normalize([{"find": "ArchiCAD", "replace": "Archicad"}])
        self.assertEqual(active_rules(data), [("ArchiCAD", "Archicad")])

    def test_accepts_enabled_as_a_synonym_for_active(self):
        data = normalize([
            {"find": "a", "replace": "1", "enabled": True},
            {"find": "b", "replace": "2", "enabled": False},
        ])
        self.assertEqual(active_rules(data), [("a", "1")])
        self.assertEqual(len(data["rules"]), 2)

    def test_active_wins_when_both_fields_are_present(self):
        data = normalize([{"find": "a", "active": False, "enabled": True}])
        self.assertEqual(active_rules(data), [])

    def test_bare_array_round_trips_into_the_wrapped_format(self):
        data = normalize([{"find": "a", "replace": "1", "enabled": True}])
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["rules"], [{"find": "a", "replace": "1", "active": True}])


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "replacements.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_round_trip_survives_a_restart(self):
        data = default_rules()
        add_rule(data, "agent teams", "Agent Teams")
        add_rule(data, "嗯嗯", "")
        set_rule_active(data, "嗯嗯", False)
        save_rules(self.path, data)

        reloaded = load_rules(self.path)
        self.assertEqual(active_rules(reloaded), [("agent teams", "Agent Teams")])
        self.assertEqual(len(reloaded["rules"]), 2)

    def test_missing_file_yields_defaults(self):
        self.assertEqual(load_rules(self.path), default_rules())

    def test_corrupt_file_does_not_raise(self):
        self.path.write_text("{broken", encoding="utf-8")
        self.assertEqual(load_rules(self.path), default_rules())

    def test_saved_file_is_readable_utf8_json(self):
        data = default_rules()
        add_rule(data, "容積率", "容積率")
        save_rules(self.path, data)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(payload["rules"][0]["find"], "容積率")

    def test_save_leaves_no_temporary_files_behind(self):
        save_rules(self.path, default_rules())
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["replacements.json"])

    def _write(self, count):
        data = default_rules()
        for index in range(count):
            add_rule(data, f"rule{index}", str(index))
        save_rules(self.path, data)
        return data

    def test_a_shrinking_save_keeps_a_backup(self):
        self._write(10)
        backup = save_rules(self.path, default_rules())
        self.assertIsNotNone(backup)
        self.assertEqual(len(load_rules(backup)["rules"]), 10)
        self.assertEqual(len(load_rules(self.path)["rules"]), 0)

    def test_deleting_a_single_rule_does_not_spam_backups(self):
        data = self._write(5)
        remove_rule(data, "rule0")
        self.assertIsNone(save_rules(self.path, data))

    def test_growing_and_unchanged_saves_make_no_backup(self):
        data = self._write(3)
        add_rule(data, "extra", "x")
        self.assertIsNone(save_rules(self.path, data))
        self.assertIsNone(save_rules(self.path, data))

    def test_first_ever_save_makes_no_backup(self):
        self.assertIsNone(save_rules(self.path, default_rules()))

    def test_backup_is_a_sibling_json_file(self):
        self._write(10)
        backup = save_rules(self.path, default_rules())
        self.assertEqual(backup.parent, self.path.parent)
        self.assertTrue(backup.name.endswith(".bak.json"))


if __name__ == "__main__":
    unittest.main()
