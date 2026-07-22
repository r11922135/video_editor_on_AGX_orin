from pathlib import Path
import copy
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_video_editor.subtitles import (  # noqa: E402
    SubtitleError,
    apply_correction_rules,
    attach_source_tokens,
    audit_correction_delivery,
    build_alignment_chunks,
    build_cues,
    correction_schema,
    create_subtitled_video,
    _local_model_snapshot,
    _validated_alignment,
    mine_correction_candidates,
    project_corrections,
    render_ass,
    render_subtitle_srt,
    validate_correction_rules,
)


class SubtitleTests(unittest.TestCase):
    def test_create_pipeline_aligns_original_text_then_projects_correction(self):
        segments = [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.0,
                "text": "LOS works.",
                "words": [
                    {"start": 0.0, "end": 0.8, "word": "LOS"},
                    {"start": 0.9, "end": 2.0, "word": "works."},
                ],
            }
        ]
        observed = {}

        def fake_align(_audio, chunks, **_kwargs):
            observed["alignment_text"] = [chunk["text"] for chunk in chunks]
            words = [
                {
                    **token,
                    "source_text": token["text"],
                    "timing_source": "test",
                }
                for chunk in chunks
                for token in chunk["source_tokens"]
            ]
            return words, {
                "status": "complete",
                "aligned_chunk_count": 1,
                "fallback_chunk_count": 0,
                "errors": [],
            }

        def fake_burn(_source, _ass, output, _log, **_kwargs):
            output.write_bytes(b"subtitled-video")

        rules = [
            {
                "rule_id": "r001",
                "scope_id": "document",
                "candidate_id": "t001",
                "original": "LOS",
                "replacement": "ROS",
                "matched_occurrences": 1,
                "replacement_observed_count": 0,
                "similarity": 0.667,
            }
        ]
        with tempfile.TemporaryDirectory() as raw, patch(
            "local_video_editor.subtitles.propose_correction_rules",
            return_value=(rules, {"status": "complete"}, "{}", []),
        ), patch(
            "local_video_editor.subtitles.align_chunks", side_effect=fake_align
        ), patch(
            "local_video_editor.subtitles.burn_ass_subtitles",
            side_effect=fake_burn,
        ):
            output = Path(raw)
            result = create_subtitled_video(
                analysis_audio=output / "analysis.wav",
                edited_video=output / "edited.mp4",
                segments=segments,
                source_title="Training",
                output_dir=output,
                summary_config={
                    "ollama_url": "http://127.0.0.1:11435",
                    "model": "qwen",
                },
                subtitle_config={
                    "aligner_model": "aligner",
                    "correction_context_tokens": 8192,
                    "correction_output_tokens": 512,
                    "correction_candidate_limit": 8,
                    "correction_rule_safety_cap": 8,
                    "alignment_chunk_seconds": 30,
                },
                video_config={"codec": "libx264", "preset": "fast", "crf": 20},
            )

            self.assertEqual(observed["alignment_text"], ["LOS works."])
            self.assertIn("ROS works.", (output / "subtitle.srt").read_text())
            self.assertEqual(segments[0]["text"], "LOS works.")
            self.assertTrue(result["all_selected_corrections_delivered"])

    def test_local_model_snapshot_resolves_mounted_cache_without_user_home(self):
        with tempfile.TemporaryDirectory() as raw:
            cache = Path(raw) / "hub"
            repository = cache / "models--Qwen--Aligner"
            snapshot = repository / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (repository / "refs").mkdir()
            (repository / "refs" / "main").write_text("abc123\n")
            with patch.dict(
                os.environ,
                {"HUGGINGFACE_HUB_CACHE": str(cache)},
                clear=False,
            ):
                self.assertEqual(
                    _local_model_snapshot("Qwen/Aligner"), str(snapshot.resolve())
                )

    def test_alignment_chunks_are_ordered_bounded_and_lossless(self):
        segments = [
            {"id": 2, "start": 42.0, "end": 55.0, "text": "  Third   part. "},
            {"id": 0, "start": 0.0, "end": 12.0, "text": "First part."},
            {"id": 1, "start": 15.0, "end": 28.0, "text": "Second part."},
        ]

        chunks = build_alignment_chunks(segments, max_seconds=30)

        self.assertEqual([chunk["id"] for chunk in chunks], ["c001", "c002"])
        self.assertEqual(chunks[0]["start"], 0.0)
        self.assertEqual(chunks[0]["end"], 28.0)
        self.assertEqual(chunks[0]["text"], "First part. Second part.")
        self.assertEqual(chunks[1]["text"], "Third part.")
        self.assertTrue(
            all(chunk["end"] - chunk["start"] <= 30 for chunk in chunks)
        )
        self.assertEqual(
            [segment["id"] for chunk in chunks for segment in chunk["segments"]],
            [0, 1, 2],
        )
        self.assertEqual(segments[0]["text"], "  Third   part. ")

    def test_candidate_mining_and_rule_validation_supports_both_scopes(self):
        chunks = build_alignment_chunks(
            [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 10.0,
                    "text": "The CubeBars motor connects over LOS.",
                },
                {
                    "id": 1,
                    "start": 40.0,
                    "end": 50.0,
                    "text": "LOS communicates with ROS controllers.",
                },
            ],
            max_seconds=30,
        )
        candidates = mine_correction_candidates(chunks, limit=20)
        los_candidate = next(
            item for item in candidates if item["original"] == "LOS"
        )
        self.assertEqual(los_candidate["occurrence_count"], 2)
        self.assertEqual(los_candidate["scope_count"], 2)

        response = {
            "document_rules": [
                {
                    "candidate_id": los_candidate["candidate_id"],
                    "replacement": "ROS",
                }
            ],
            "local_rules": [
                {
                    "scope_id": "c001",
                    "original": "CubeBars",
                    "replacement": "CubeMars",
                }
            ],
        }

        rules, report = validate_correction_rules(
            response, chunks, candidates, max_rules=10
        )

        self.assertEqual(
            [
                (rule["scope_id"], rule["original"], rule["replacement"])
                for rule in rules
            ],
            [
                ("document", "LOS", "ROS"),
                ("c001", "CubeBars", "CubeMars"),
            ],
        )
        self.assertEqual(rules[0]["matched_occurrences"], 2)
        self.assertEqual(report["accepted_document_rule_count"], 1)
        self.assertEqual(report["accepted_local_rule_count"], 1)

    def test_candidate_mining_supports_spaced_names_and_acronyms(self):
        chunks = build_alignment_chunks(
            [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 8.0,
                    "text": "Cube Bars uses R O S. Cube Bars teaches R O S.",
                }
            ],
            max_seconds=30,
        )

        candidates = mine_correction_candidates(chunks, limit=30)
        by_original = {item["original"]: item for item in candidates}

        self.assertEqual(by_original["Cube Bars"]["occurrence_count"], 2)
        self.assertEqual(by_original["R O S"]["occurrence_count"], 2)
        response = {
            "document_rules": [
                {
                    "candidate_id": by_original["R O S"]["candidate_id"],
                    "replacement": "ROS",
                }
            ],
            "local_rules": [],
        }
        rules, report = validate_correction_rules(
            response, chunks, candidates, max_rules=10
        )
        self.assertEqual(
            (rules[0]["original"], rules[0]["replacement"]),
            ("R O S", "ROS"),
        )
        self.assertEqual(report["accepted_document_rule_count"], 1)

    def test_rule_validation_rejects_unknown_or_unsafe_corrections(self):
        chunks = build_alignment_chunks(
            [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 8.0,
                    "text": "Use C++, 中文, and VLA here.",
                }
            ],
            max_seconds=30,
        )
        candidates = mine_correction_candidates(chunks, limit=20)
        response = {
            "document_rules": [
                {"candidate_id": "missing", "replacement": "ROS"},
            ],
            "local_rules": [
                {
                    "scope_id": "c001",
                    "original": "C++",
                    "replacement": "C#",
                },
                {
                    "scope_id": "c001",
                    "original": "中文",
                    "replacement": "日文",
                },
                {
                    "scope_id": "c001",
                    "original": "VLA",
                    "replacement": "VR",
                },
                {
                    "scope_id": "c999",
                    "original": "motor",
                    "replacement": "rotor",
                },
            ],
        }

        rules, report = validate_correction_rules(
            response, chunks, candidates, max_rules=10
        )

        self.assertEqual(rules, [])
        self.assertEqual(report["rejected_rule_count"], 5)
        self.assertEqual(report["rejection_reasons"]["unknown_candidate"], 1)
        self.assertEqual(report["rejection_reasons"]["unknown_scope"], 1)

    def test_local_rule_rejects_ordinary_semantic_rewrite(self):
        chunks = build_alignment_chunks(
            [{"id": 0, "start": 0.0, "end": 2.0, "text": "The motor turns."}],
            max_seconds=30,
        )
        response = {
            "document_rules": [],
            "local_rules": [
                {
                    "scope_id": "c001",
                    "original": "motor",
                    "replacement": "rotor",
                }
            ],
        }
        rules, report = validate_correction_rules(
            response, chunks, [], max_rules=10
        )
        self.assertEqual(rules, [])
        self.assertEqual(report["rejection_reasons"]["unsafe_local_scope"], 1)

    def test_document_rule_accepts_observed_acronym_with_asr_suffix(self):
        chunks = build_alignment_chunks(
            [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 4.0,
                    "text": "ROS works while LOST starts. LOST stops.",
                }
            ],
            max_seconds=30,
        )
        candidates = mine_correction_candidates(chunks, limit=20)
        lost = next(item for item in candidates if item["original"] == "LOST")
        rules, _report = validate_correction_rules(
            {
                "document_rules": [
                    {"candidate_id": lost["candidate_id"], "replacement": "ROS"}
                ],
                "local_rules": [],
            },
            chunks,
            candidates,
            max_rules=10,
        )
        self.assertEqual(
            (rules[0]["original"], rules[0]["replacement"]), ("LOST", "ROS")
        )

    def test_document_rule_balances_unseen_technical_replacements(self):
        chunks = build_alignment_chunks(
            [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 4.0,
                    "text": "RVs uses RVs, TI2 starts TI2, and LOS4 starts LOS4.",
                }
            ],
            max_seconds=30,
        )
        candidates = mine_correction_candidates(chunks, limit=20)
        by_original = {item["original"]: item for item in candidates}
        rules, report = validate_correction_rules(
            {
                "document_rules": [
                    {
                        "candidate_id": by_original["RVs"]["candidate_id"],
                        "replacement": "RViz",
                    },
                    {
                        "candidate_id": by_original["LOS4"]["candidate_id"],
                        "replacement": "ROS1",
                    },
                    {
                        "candidate_id": by_original["TI2"]["candidate_id"],
                        "replacement": "T265",
                    },
                ],
                "local_rules": [],
            },
            chunks,
            candidates,
            max_rules=10,
        )
        self.assertEqual(
            [(rule["original"], rule["replacement"]) for rule in rules],
            [("RVs", "RViz")],
        )
        self.assertEqual(report["rejection_reasons"]["low_similarity"], 2)

    def test_document_rule_crosses_chunks_and_local_rule_takes_precedence(self):
        chunks = build_alignment_chunks(
            [
                {"id": 0, "start": 0.0, "end": 5.0, "text": "LOS starts."},
                {"id": 1, "start": 40.0, "end": 45.0, "text": "LOS ends."},
            ],
            max_seconds=30,
        )
        original = copy.deepcopy(chunks)
        rules = [
            {
                "rule_id": "r001",
                "scope_id": "document",
                "original": "LOS",
                "replacement": "ROS",
                "matched_occurrences": 2,
            },
            {
                "rule_id": "r002",
                "scope_id": "c002",
                "original": "LOS",
                "replacement": "RQS",
                "matched_occurrences": 1,
            },
        ]

        corrected = apply_correction_rules(chunks, rules)

        self.assertEqual(
            [chunk["text"] for chunk in corrected],
            ["ROS starts.", "RQS ends."],
        )
        self.assertEqual(corrected[0]["occurrences"][0]["rule_id"], "r001")
        self.assertEqual(corrected[1]["occurrences"][0]["rule_id"], "r002")
        self.assertEqual(chunks, original)

    def test_apply_rules_does_not_mutate_canonical_chunks_or_partial_words(self):
        chunks = [
            {
                "id": "c001",
                "start": 0.0,
                "end": 2.0,
                "text": "LOS LOS2 LOS.",
                "segments": [{"text": "LOS LOS2 LOS."}],
            }
        ]
        original = copy.deepcopy(chunks)
        rules = [
            {
                "rule_id": "r001",
                "scope_id": "document",
                "original": "LOS",
                "replacement": "ROS",
                "matched_occurrences": 2,
            }
        ]

        corrected = apply_correction_rules(chunks, rules)

        self.assertEqual(corrected[0]["text"], "ROS LOS2 ROS.")
        self.assertEqual(corrected[0]["applied_rules"][0]["applied_occurrences"], 2)
        self.assertEqual(chunks, original)
        self.assertIsNot(corrected[0], chunks[0])

    def test_document_rule_does_not_modify_compound_identifiers(self):
        chunks = build_alignment_chunks(
            [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.0,
                    "text": "LOS LOS.com LOS-based LOS.",
                }
            ],
            max_seconds=30,
        )
        corrected = apply_correction_rules(
            chunks,
            [
                {
                    "rule_id": "r001",
                    "scope_id": "document",
                    "original": "LOS",
                    "replacement": "ROS",
                    "matched_occurrences": 2,
                }
            ],
        )
        self.assertEqual(corrected[0]["text"], "ROS LOS.com LOS-based ROS.")
        self.assertEqual(len(corrected[0]["occurrences"]), 2)

    def test_source_ids_are_unique_when_external_segment_ids_repeat(self):
        chunks = build_alignment_chunks(
            [
                {"id": 7, "start": 0.0, "end": 1.0, "text": "First."},
                {"id": 7, "start": 2.0, "end": 3.0, "text": "Second."},
            ],
            max_seconds=30,
        )
        source_ids = [
            token["source_id"]
            for chunk in chunks
            for token in chunk["source_tokens"]
        ]
        self.assertEqual(len(source_ids), len(set(source_ids)))

    def test_correction_schema_limits_scopes_and_rule_count(self):
        schema = correction_schema(["c001", "c002"], 7)
        document_rules = schema["properties"]["document_rules"]
        local_rules = schema["properties"]["local_rules"]
        self.assertEqual(document_rules["maxItems"], 7)
        self.assertEqual(local_rules["maxItems"], 7)
        self.assertEqual(
            local_rules["items"]["properties"]["scope_id"]["enum"],
            ["c001", "c002"],
        )
        self.assertEqual(
            set(schema["required"]), {"document_rules", "local_rules"}
        )
        self.assertFalse(schema["additionalProperties"])

    def test_source_token_alignment_ignores_case_and_punctuation_only(self):
        chunk = build_alignment_chunks(
            [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello, WORLD!",
                }
            ],
            max_seconds=30,
        )[0]
        aligned = [
            {"start": 0.0, "end": 0.4, "text": "hello"},
            {"start": 0.5, "end": 0.9, "text": "world"},
        ]

        result = attach_source_tokens(chunk, aligned)

        self.assertEqual(
            [item["source_text"] for item in result], ["Hello,", "WORLD!"]
        )
        self.assertEqual(
            [item["timing_source"] for item in result],
            ["forced_aligner"] * 2,
        )
        self.assertEqual(result[1]["end"], 0.9)

    def test_alignment_clamps_small_boundary_overrun_and_timestamp_jitter(self):
        result = _validated_alignment(
            [
                {"start": 1.0, "end": 1.2, "text": "first"},
                {"start": 0.97, "end": 1.4, "text": "second"},
                {"start": 5.8, "end": 6.4, "text": "last"},
            ],
            audio_seconds=5.0,
        )
        self.assertEqual(result[1]["start"], 1.0)
        self.assertEqual((result[-1]["start"], result[-1]["end"]), (5.0, 5.0))

    def test_one_to_many_replacement_projects_over_original_timing(self):
        chunks = build_alignment_chunks(
            [
                {
                    "id": 0,
                    "start": 2.0,
                    "end": 4.0,
                    "text": "LOS works.",
                }
            ],
            max_seconds=30,
        )
        corrected = apply_correction_rules(
            chunks,
            [
                {
                    "rule_id": "r001",
                    "scope_id": "document",
                    "original": "LOS",
                    "replacement": "ROS 2",
                    "matched_occurrences": 1,
                }
            ],
        )
        timed_words = [
            {
                **token,
                "source_text": token["text"],
                "timing_source": "forced_aligner",
            }
            for token in chunks[0]["source_tokens"]
        ]

        display = project_corrections(timed_words, corrected)

        self.assertEqual([item["text"] for item in display], ["ROS 2", "works."])
        self.assertEqual(display[0]["start"], timed_words[0]["start"])
        self.assertEqual(display[0]["end"], timed_words[0]["end"])
        self.assertEqual(display[0]["source_text"], "LOS")
        self.assertEqual(display[0]["correction_ids"], ["c001:o0001"])

    def test_fallback_timeline_keeps_corrected_display_text(self):
        chunks = build_alignment_chunks(
            [
                {
                    "id": 0,
                    "start": 5.0,
                    "end": 7.0,
                    "text": "LOS navigation.",
                }
            ],
            max_seconds=30,
        )
        corrected = apply_correction_rules(
            chunks,
            [
                {
                    "rule_id": "r001",
                    "scope_id": "document",
                    "original": "LOS",
                    "replacement": "ROS",
                    "matched_occurrences": 1,
                }
            ],
        )
        fallback_words = [
            {
                **token,
                "source_text": token["text"],
                "timing_source": "whisper_fallback",
            }
            for token in chunks[0]["source_tokens"]
        ]

        display = project_corrections(fallback_words, corrected)

        self.assertEqual(display[0]["text"], "ROS")
        self.assertEqual(display[0]["source_text"], "LOS")
        self.assertEqual(display[0]["timing_source"], "whisper_fallback")
        self.assertEqual(display[0]["start"], fallback_words[0]["start"])
        self.assertEqual(display[0]["end"], fallback_words[0]["end"])

    def test_projected_multiword_timing_uses_the_full_source_span(self):
        chunks = build_alignment_chunks(
            [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 4.0,
                    "text": "Cube Bars works.",
                }
            ],
            max_seconds=30,
        )
        corrected = apply_correction_rules(
            chunks,
            [
                {
                    "rule_id": "r001",
                    "scope_id": "document",
                    "original": "Cube Bars",
                    "replacement": "CubeMars",
                    "matched_occurrences": 1,
                }
            ],
        )
        timed_words = [
            {
                **token,
                "source_text": token["text"],
                "start": start,
                "end": end,
            }
            for token, start, end in zip(
                chunks[0]["source_tokens"],
                [2.0, 1.0, 3.0],
                [3.0, 4.0, 3.5],
            )
        ]
        display = project_corrections(timed_words, corrected)
        self.assertEqual(display[0]["text"], "CubeMars")
        self.assertEqual((display[0]["start"], display[0]["end"]), (1.0, 4.0))

    def test_delivery_audit_accepts_complete_projection_and_detects_drop(self):
        chunks = build_alignment_chunks(
            [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.0,
                    "text": "LOS works.",
                }
            ],
            max_seconds=30,
        )
        rules = [
            {
                "rule_id": "r001",
                "scope_id": "document",
                "original": "LOS",
                "replacement": "ROS",
                "matched_occurrences": 1,
            }
        ]
        corrected = apply_correction_rules(chunks, rules)
        timed_words = [
            {
                **token,
                "source_text": token["text"],
                "timing_source": "whisper_fallback",
            }
            for token in chunks[0]["source_tokens"]
        ]
        display = project_corrections(timed_words, corrected)
        cues = build_cues(display)

        audit = audit_correction_delivery(corrected, display, cues, rules)

        self.assertTrue(audit["all_selected_corrections_delivered"])
        self.assertEqual(audit["delivered_occurrence_count"], 1)
        dropped = copy.deepcopy(cues)
        dropped[0]["correction_ids"] = []
        with self.assertRaisesRegex(SubtitleError, "not delivered exactly once"):
            audit_correction_delivery(corrected, display, dropped, rules)

    def test_cues_obey_readability_bounds_without_overlaps(self):
        words = []
        for index in range(30):
            start = index * 0.32
            words.append(
                {
                    "start": start,
                    "end": start + 0.25,
                    "text": f"word{index}",
                }
            )
        words[5]["text"] = "sentence."

        cues = build_cues(words)

        self.assertGreater(len(cues), 1)
        self.assertTrue(all(cue["end"] > cue["start"] for cue in cues))
        self.assertTrue(all(cue["end"] - cue["start"] <= 6.0 for cue in cues))
        self.assertTrue(all(len(cue["text"]) <= 78 for cue in cues))
        self.assertTrue(
            all(left["end"] <= right["start"] for left, right in zip(cues, cues[1:]))
        )
        self.assertIn("word0 word1", cues[0]["text"])
        self.assertTrue(cues[0]["text"].endswith("sentence."))

        srt = render_subtitle_srt(cues)
        self.assertIn("00:00:00,000 -->", srt)
        text_lines = [
            line
            for line in srt.splitlines()
            if line and not line.isdigit() and " --> " not in line
        ]
        self.assertTrue(all(len(line) <= 42 for line in text_lines))

    def test_cues_normalize_overlapping_and_abnormally_long_fallback_words(self):
        cues = build_cues(
            [
                {"start": 1.0, "end": 15.0, "text": "so"},
                {"start": 2.0, "end": 2.4, "text": "the"},
                {"start": 2.4, "end": 11.0, "text": "motor."},
                {"start": 3.0, "end": 3.5, "text": "Next."},
            ]
        )
        self.assertTrue(all(cue["end"] > cue["start"] for cue in cues))
        self.assertTrue(all(cue["end"] - cue["start"] <= 6.0 for cue in cues))
        self.assertTrue(
            all(left["end"] <= right["start"] for left, right in zip(cues, cues[1:]))
        )

    def test_cues_preserve_source_order_during_small_timestamp_jitter(self):
        cues = build_cues(
            [
                {"start": 1.00, "end": 1.20, "text": "first"},
                {"start": 0.97, "end": 1.25, "text": "second"},
            ]
        )
        self.assertEqual(" ".join(cue["text"] for cue in cues), "first second")

    def test_ass_uses_readable_style_and_escapes_override_characters(self):
        ass = render_ass(
            [
                {
                    "start": 1.25,
                    "end": 3.5,
                    "text": (
                        r"Path C:\robot\config {draft} contains a very long "
                        "technical phrase for line balancing"
                    ),
                }
            ]
        )

        self.assertIn("PlayResX: 1920", ass)
        self.assertIn("PlayResY: 1080", ass)
        self.assertIn("Style: Default,DejaVu Sans,52", ass)
        self.assertIn("Dialogue: 0,0:00:01.25,0:00:03.50", ass)
        self.assertIn("C:\\\u2060robot\\\u2060config", ass)
        self.assertIn(r"\{draft\}", ass)
        self.assertIn(r"\N", ass)


if __name__ == "__main__":
    unittest.main()
