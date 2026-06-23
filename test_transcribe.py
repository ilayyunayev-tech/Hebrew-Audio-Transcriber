import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import transcribe


class TranscribeTests(unittest.TestCase):
    def test_missing_file(self):
        with patch("sys.stderr"):
            self.assertEqual(transcribe.main(["missing.mp3"]), 1)

    def test_unsupported_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.txt"
            path.touch()
            with patch("sys.stderr"):
                self.assertEqual(transcribe.main([str(path)]), 1)

    def test_transcribe_audio_consumes_generator(self):
        model = Mock()
        model.transcribe.return_value = (
            iter(
                [
                    SimpleNamespace(start=0.0, end=1.0, text=" שלום "),
                    SimpleNamespace(start=1.0, end=2.0, text=" לכולם "),
                ]
            ),
            SimpleNamespace(language="he", duration=2.0),
        )

        with patch("sys.stderr"):
            text, segments = transcribe.transcribe_audio(
                model, Path("audio.mp3")
            )

        self.assertEqual(text, "שלום לכולם")
        self.assertEqual(len(segments), 2)
        model.transcribe.assert_called_once_with(
            "audio.mp3",
            language="he",
            beam_size=1,
            best_of=1,
            vad_filter=True,
        )

    def test_progress_reaches_100_percent(self):
        model = Mock()
        model.transcribe.return_value = (
            iter(
                [
                    SimpleNamespace(start=0.0, end=2.5, text="חצי"),
                    SimpleNamespace(start=2.5, end=5.0, text="סוף"),
                ]
            ),
            SimpleNamespace(language="he", duration=5.0),
        )

        with patch("sys.stderr") as stderr:
            transcribe.transcribe_audio(model, Path("audio.mp3"))

        progress_output = "".join(
            call.args[0] for call in stderr.write.call_args_list if call.args
        )
        self.assertIn(" 50%", progress_output)
        self.assertIn("100%", progress_output)

    def test_save_txt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "transcript.txt"
            transcribe.save_txt("שלום", output)
            self.assertEqual(output.read_text(encoding="utf-8"), "שלום\n")

    def test_format_timeline(self):
        segments = [
            SimpleNamespace(start=1.2, end=4.6, text=" first "),
            SimpleNamespace(start=4.6, end=5.0, text=" "),
            SimpleNamespace(start=65.0, end=3661.0, text=" second "),
        ]

        self.assertEqual(
            transcribe.format_timeline(segments),
            (
                "[00:00:01 - 00:00:05] first\n"
                "[00:01:05 - 01:01:01] second"
            ),
        )

    def test_save_timeline(self):
        segments = [
            SimpleNamespace(start=0.0, end=2.0, text=" hello "),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "transcript.txt"
            transcribe.save_timeline(segments, output)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "[00:00:00 - 00:00:02] hello\n",
            )

    def test_save_srt(self):
        segments = [
            SimpleNamespace(start=0.0, end=1.25, text=" שלום "),
            SimpleNamespace(start=1.25, end=2.0, text=" "),
            SimpleNamespace(start=61.5, end=62.0, text=" עולם "),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "transcript.srt"
            transcribe.save_srt(segments, output)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                (
                    "1\n00:00:00,000 --> 00:00:01,250\nשלום\n\n"
                    "2\n00:01:01,500 --> 00:01:02,000\nעולם\n"
                ),
            )

    @patch("transcribe.transcribe_audio")
    @patch("transcribe.load_model")
    def test_main_prints_transcript(self, load_model, transcribe_audio):
        load_model.return_value = object()
        transcribe_audio.return_value = ("שלום", [])

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.mp3"
            audio.touch()
            with patch("builtins.print") as print_mock:
                result = transcribe.main([str(audio)])

        self.assertEqual(result, 0)
        print_mock.assert_called_once_with("שלום")

    @patch("transcribe.transcribe_audio")
    @patch("transcribe.load_model")
    def test_main_prints_timeline(self, load_model, transcribe_audio):
        load_model.return_value = object()
        segments = [
            SimpleNamespace(start=0.0, end=2.0, text=" hello "),
        ]
        transcribe_audio.return_value = ("hello", segments)

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.mp3"
            audio.touch()
            with patch("builtins.print") as print_mock:
                result = transcribe.main([str(audio), "--timeline"])

        self.assertEqual(result, 0)
        print_mock.assert_called_once_with(
            "[00:00:00 - 00:00:02] hello"
        )

    @patch("transcribe.transcribe_audio", side_effect=KeyboardInterrupt)
    @patch("transcribe.load_model")
    def test_ctrl_c_cancels_cleanly(self, load_model, transcribe_audio):
        load_model.return_value = object()

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.mp3"
            audio.touch()
            with patch("sys.stderr") as stderr:
                result = transcribe.main([str(audio)])

        self.assertEqual(result, 130)
        output = "".join(
            call.args[0] for call in stderr.write.call_args_list if call.args
        )
        self.assertIn("Transcription cancelled.", output)


if __name__ == "__main__":
    unittest.main()
