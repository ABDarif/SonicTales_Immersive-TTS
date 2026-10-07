import json

import pytest

from app.models import NARRATOR
from app.pdf_extract import book_from_text
from app.script_builder import LLMScriptBuilder, build_script, chunk_paragraphs

from .conftest import STORY


def script_for(text: str):
    return build_script(book_from_text(text, "T"), "heuristic")


def lines(script, chapter=0):
    return [(s.speaker, s.emotion, s.text) for sc in script.chapters[chapter].scenes for s in sc.segments]


def test_dialogue_is_split_from_narration_and_attributed():
    segs = lines(script_for(STORY))
    assert ("Mary", "neutral", "You're late,") in segs
    assert ("Tom", "neutral", "The storm,") in segs
    assert ("Tom", "neutral", "The road was flooded.") in segs  # same paragraph, tag in the middle
    assert any(sp == NARRATOR and "said Mary" in t for sp, _, t in segs), "speech tags stay with the narrator"


def test_unattributed_reply_alternates_speaker():
    segs = lines(script_for(STORY))
    assert ("Mary", "neutral", "Of course it was.") in segs


def test_pronoun_resolution_and_emotion_from_verb():
    segs = lines(script_for(STORY))
    assert ("Mary", "whisper", "I was worried.") in segs


def test_emotion_from_adverb_and_exclamation():
    script = script_for('"Run!" Tom shouted.\n\n"Fine," said Tom sadly.\n\n"Wow!" said Tom.')
    emotions = [e for _, e, _ in lines(script)]
    assert emotions[0] == "angry" and "sad" in emotions and "excited" in emotions


def test_genders_inferred():
    script = script_for(STORY)
    assert script.characters["Mary"].gender == "female"
    assert script.characters["Tom"].gender == "male"


def test_gender_from_pronouns_for_unknown_names():
    text = (
        'Zorblax grabbed his coat and left. "Bye," said Zorblax.\n\nQuill smiled as she waved. "Goodbye," said Quill.'
    )
    script = script_for(text)
    assert script.characters["Zorblax"].gender == "male"
    assert script.characters["Quill"].gender == "female"


def test_first_person_speaker_is_narrator():
    script = script_for('"Hello," I said.')
    assert lines(script)[0][0] == NARRATOR


def test_stopwords_are_not_names():
    script = script_for('"Hello," said The old man.')
    assert "The" not in script.characters


def test_ambience_follows_the_story():
    script = script_for(STORY)
    first, second = script.chapters[0].scenes
    assert {"rain", "night", "fire"} <= set(first.ambience)
    assert "rain" not in second.ambience and "night" not in second.ambience
    assert "birds" in second.ambience
    assert script.chapters[1].scenes[0].ambience == ["wind", "ocean"]


def test_scene_break_marker_starts_new_scene():
    script = script_for("It rained.\n\n* * *\n\nStill it rained all day.")
    assert len(script.chapters[0].scenes) == 2


def test_rain_stopping_ends_rain_ambience():
    text = "Rain fell on the roof.\n\nThe rain stopped at last.\n\nThey went home quietly."
    scenes = script_for(text).chapters[0].scenes
    assert scenes[0].ambience == ["rain"]
    assert all("rain" not in sc.ambience for sc in scenes[1:])


def test_night_persists_until_morning():
    text = "It was midnight.\n\n" + "\n\n".join(f"They walked on, step {i}." for i in range(8)) + "\n\nMorning arrived."
    scenes = script_for(text).chapters[0].scenes
    assert scenes[0].ambience == ["night"]
    assert scenes[-1].ambience == []


def test_pause_after_set_and_longer_at_paragraph_end():
    segs = [s for sc in script_for(STORY).chapters[0].scenes for s in sc.segments]
    assert all(s.pause_after is not None for s in segs)
    assert max(s.pause_after for s in segs) >= 0.7


def test_plain_text_without_dialogue_is_all_narrator():
    script = script_for("Just a calm description of a quiet field.")
    assert {s for s, _, _ in lines(script)} == {NARRATOR}
    assert script.characters == {}


# ---------------------------------------------------------------- LLM builder


def llm_response(scenes, characters=()):
    return json.dumps({"characters": list(characters), "scenes": scenes})


def test_llm_builder_uses_valid_output():
    text = '"Hi," said Ann.\n\nIt was raining.'
    reply = llm_response(
        [
            {
                "ambience": ["rain", "bogus"],
                "segments": [
                    {"speaker": "Ann", "text": "Hi,", "emotion": "happy"},
                    {"speaker": "Narrator", "text": "said Ann. It was raining.", "emotion": "weird"},
                ],
            }
        ],
        [{"name": "Ann", "gender": "female"}],
    )
    script = build_script(book_from_text(text), "llm", complete=lambda s, u: f"```json\n{reply}\n```")
    scene = script.chapters[0].scenes[0]
    assert scene.ambience == ["rain"], "unknown ambience tags are dropped"
    assert scene.segments[0].emotion == "happy" and scene.segments[0].is_dialogue
    assert scene.segments[1].emotion == "neutral", "unknown emotions are coerced"
    assert script.characters["Ann"].gender == "female"


def test_llm_builder_falls_back_on_bad_json():
    script = build_script(book_from_text(STORY), "llm", complete=lambda s, u: "I cannot do that")
    assert script.characters["Mary"].gender == "female"  # produced by the heuristic fallback
    assert script.chapters[0].scenes


def test_llm_builder_falls_back_when_text_is_dropped():
    # The model "summarises": covers far less than the source text, so it must be rejected.
    reply = llm_response(
        [{"ambience": [], "segments": [{"speaker": "Narrator", "text": "Stuff happened.", "emotion": "neutral"}]}]
    )
    builder = LLMScriptBuilder(lambda s, u: reply)
    chapter = builder.build_chapter(1, book_from_text(STORY).chapters[0])
    assert builder.fallback_chunks == 2  # the chapter has a scene break, so it is sent as two chunks
    assert any(s.speaker == "Mary" for sc in chapter.scenes for s in sc.segments)


def test_llm_builder_falls_back_when_client_raises():
    def boom(system, user):
        raise ConnectionError("network down")

    script = build_script(book_from_text(STORY), "llm", complete=boom)
    assert script.chapters[0].scenes


def test_llm_receives_known_characters_on_later_chunks():
    seen = []
    first = llm_response(
        [
            {
                "ambience": [],
                "segments": [{"speaker": "Narrator", "text": "Ann waved to the crowd outside.", "emotion": "neutral"}],
            }
        ],
        [{"name": "Ann", "gender": "female"}],
    )

    def complete(system, user):
        seen.append(user)
        return first

    builder = LLMScriptBuilder(complete)
    builder.build_chapter(1, book_from_text("Ann waved to the crowd outside.").chapters[0])
    builder.build_chapter(2, book_from_text("Ann waved to the crowd outside.").chapters[0])
    assert "none yet" in seen[0] and "Ann (female)" in seen[1]


def test_auto_mode_without_key_uses_heuristic(settings):
    script = build_script(book_from_text(STORY), "auto", settings)
    assert script.characters["Tom"].gender == "male"


def test_llm_mode_without_key_is_an_error(settings):
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_script(book_from_text(STORY), "llm", settings)


def test_chunking_respects_limit_and_scene_breaks():
    chunks = chunk_paragraphs(["a" * 3000, "b" * 3000, "<<SCENE_BREAK>>", "c"], max_chars=5000)
    assert [len(c) for c in chunks] == [1, 1, 2]


PRIDE_AND_PREJUDICE = """Chapter 1

“My dear Mr. Bennet,” said his lady to him one day, “have you heard that Netherfield Park is let at last?”

Mr. Bennet replied that he had not.

“But it is,” returned she; “for Mrs. Long has just been here, and she told me all about it.”

Mr. Bennet made no answer.

“Do you not want to know who has taken it?” cried his wife impatiently.

“You want to tell me, and I have no objection to hearing it.”

This was invitation enough.

“Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man.”

“What is his name?”

“Bingley.”

“Is he married or single?”
"""


def test_real_prose_two_person_conversation_keeps_two_voices():
    """Regression: descriptor attributions ('said his lady'), 'returned she', reported speech and
    unattributed replies must resolve to two alternating characters, not one 'Stranger'."""
    segs = [(sp, t) for sp, _, t in lines(script_for(PRIDE_AND_PREJUDICE)) if sp != NARRATOR]
    speakers = [sp for sp, _ in segs]
    assert set(speakers) == {"Woman", "Mr. Bennet"}
    assert speakers == ["Woman"] * 5 + ["Mr. Bennet", "Woman", "Mr. Bennet", "Woman", "Mr. Bennet"]
    script = script_for(PRIDE_AND_PREJUDICE)
    assert script.characters["Woman"].gender == "female" and script.characters["Mr. Bennet"].gender == "male"


def test_role_nouns_resolve_to_one_character():
    script = script_for('"Hi," said the old man.\n\n"Well," replied the gentleman.')
    assert {s for s, _, _ in lines(script) if s != NARRATOR} == {"Man"}


def test_one_narration_paragraph_does_not_break_alternation():
    text = '"Hello," said Tom.\n\n"Hi," said Mary.\n\nThe fire crackled.\n\n"Cold tonight."\n\n"Very."'
    speakers = [s for s, _, _ in lines(script_for(text)) if s != NARRATOR]
    assert speakers == ["Tom", "Mary", "Tom", "Mary"]
