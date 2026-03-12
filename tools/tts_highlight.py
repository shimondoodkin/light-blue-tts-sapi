"""
Qt application that speaks Hebrew text via SAPI and highlights
the current word using word boundary events from SpVoice.Status.

Requirements:
    pip install PyQt5 pywin32
"""

import sys
import html
import traceback

# IMPORTANT: Initialize COM and trigger SAPI/onnxruntime DLL loading BEFORE
# importing PyQt5.  PyQt5 bundles its own (incompatible) msvcp140.dll in
# Qt5/bin, which poisons the process and prevents onnxruntime.dll from loading.
# By loading SAPI first, onnxruntime.dll gets linked against the correct system
# VC++ runtime DLLs.
import pythoncom
pythoncom.CoInitialize()
import win32com.client
_warmup_voice = win32com.client.Dispatch("SAPI.SpVoice")
# Enumerate voices to trigger DLL loading in LightBlue engines
_voices = _warmup_voice.GetVoices()
for _i in range(_voices.Count):
    _desc = _voices.Item(_i).GetDescription()
    if "LightBlue" in _desc:
        _warmup_voice.Voice = _voices.Item(_i)
        # Speak empty string to trigger model warm-up
        _warmup_voice.Speak("", 1)  # SVSFlagsAsync
        break
del _warmup_voice, _voices

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QComboBox,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont


SAMPLE_TEXT = "יש לי 3 חתולים ו-15 כלבים. הם אוהבים לשחק בגינה כל יום."


class TTSHighlightApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LightBlue TTS - Word Highlight")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        # --- SAPI voice (COM already initialized at module level) ---
        self.voice = win32com.client.Dispatch("SAPI.SpVoice")
        # Ensure word boundary events are enabled
        # SVEWordBoundary = 0x20 (bit 5), SVEAllEvents = 0x7FFFE
        try:
            self.voice.EventInterests = 0x7FFFE  # SVEAllEvents
        except Exception as e:
            print(f"Could not set EventInterests: {e}")
        self.speaking = False

        # --- Layout ---
        layout = QVBoxLayout(self)

        # Voice selector
        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(350)
        self._populate_voices()
        voice_layout.addWidget(self.voice_combo, 1)
        layout.addLayout(voice_layout)

        # Text input
        layout.addWidget(QLabel("Text to speak:"))
        self.input_edit = QTextEdit()
        self.input_edit.setPlainText(SAMPLE_TEXT)
        self.input_edit.setMaximumHeight(100)
        self.input_edit.setFont(QFont("Segoe UI", 14))
        layout.addWidget(self.input_edit)

        # Buttons
        btn_layout = QHBoxLayout()
        self.speak_btn = QPushButton("Speak")
        self.speak_btn.setFont(QFont("Segoe UI", 12))
        self.speak_btn.clicked.connect(self.on_speak)
        btn_layout.addWidget(self.speak_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFont(QFont("Segoe UI", 12))
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        # Rendered text with highlighting
        layout.addWidget(QLabel("Spoken text:"))
        self.display_label = QLabel()
        self.display_label.setWordWrap(True)
        self.display_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.display_label.setLayoutDirection(Qt.RightToLeft)
        self.display_label.setFont(QFont("Segoe UI", 18))
        self.display_label.setTextFormat(Qt.RichText)
        self.display_label.setStyleSheet(
            "QLabel { background: white; border: 1px solid #ccc; "
            "padding: 16px; border-radius: 6px; }"
        )
        self.display_label.setMinimumHeight(120)
        layout.addWidget(self.display_label, 1)

        # Debug / Status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-family: monospace;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Poll timer for word boundary
        self.timer = QTimer()
        self.timer.timeout.connect(self.poll_word_position)

        # Current state
        self.current_text = ""
        self.last_pos = -1
        self.last_len = 0

    def _populate_voices(self):
        """Fill the voice combo with available SAPI voices, selecting first LightBlue one."""
        voices = self.voice.GetVoices()
        select_idx = 0
        for i in range(voices.Count):
            v = voices.Item(i)
            desc = v.GetDescription()
            self.voice_combo.addItem(desc, i)
            if "LightBlue" in desc and "8 CPU" in desc and select_idx == 0:
                select_idx = self.voice_combo.count() - 1
        if select_idx:
            self.voice_combo.setCurrentIndex(select_idx)

    def _set_voice(self):
        """Apply the currently selected voice."""
        idx = self.voice_combo.currentData()
        if idx is not None:
            voices = self.voice.GetVoices()
            self.voice.Voice = voices.Item(idx)

    def on_speak(self):
        self.current_text = self.input_edit.toPlainText().strip()
        if not self.current_text:
            return

        self._set_voice()
        self._render_text(-1, 0)

        # Purge any pending speech, then speak async
        # SVSFlagsAsync=1, SVSFPurgeBeforeSpeak=2
        self.voice.Speak(self.current_text, 1 | 2)
        self.speaking = True
        self.last_pos = -1
        self.last_len = 0

        self.speak_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Speaking...")
        self.timer.start(30)

    def on_stop(self):
        # Purge speech
        try:
            self.voice.Speak("", 2)  # SVSFPurgeBeforeSpeak
        except Exception:
            pass
        self._finish()

    def _finish(self):
        self.timer.stop()
        self.speaking = False
        self.speak_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Done.")
        self._render_text(-1, 0)

    def poll_word_position(self):
        """Called by QTimer to read current word position from SAPI."""
        try:
            pythoncom.PumpWaitingMessages()
            status = self.voice.Status
            running = status.RunningState  # 0=done, 1=done, 2=speaking

            if running != 2:
                self._finish()
                return

            pos = status.InputWordPosition
            length = status.InputWordLength

            if pos != self.last_pos or length != self.last_len:
                self.last_pos = pos
                self.last_len = length

                # Debug: print raw values to console and status bar
                text_len = len(self.current_text)
                debug = f"pos={pos} len={length} text_len={text_len}"
                print(debug)

                # Try to figure out correct char position.
                # SAPI may report byte offset in UTF-16 (2 bytes per BMP char)
                # or it may report char index directly — depends on the engine.
                char_pos, char_len = self._resolve_position(pos, length)
                debug += f" -> char_pos={char_pos} char_len={char_len}"

                if 0 <= char_pos < text_len and char_len > 0:
                    word = self.current_text[char_pos:char_pos + char_len]
                    debug += f" word='{word}'"
                    self._render_text(char_pos, char_len)
                else:
                    debug += " (out of bounds, skipping)"

                self.status_label.setText(debug)

        except Exception as e:
            tb = traceback.format_exc()
            print(f"Poll error: {tb}")
            self.status_label.setText(f"Error: {e}")
            # Don't stop on error — keep trying
            # self._finish()

    def _resolve_position(self, raw_pos: int, raw_len: int):
        """
        Convert SAPI position to Python string index.
        SAPI engines may report positions as:
          1. Character index (direct) — if pos < len(text), likely this
          2. UTF-16 byte offset — if pos >= len(text), divide by 2
        We try direct first, then halved.
        """
        text_len = len(self.current_text)

        # Direct character index
        if raw_pos < text_len and raw_len > 0:
            end = raw_pos + raw_len
            if end <= text_len:
                return raw_pos, raw_len
            else:
                # Length overshoots — clamp it
                return raw_pos, text_len - raw_pos

        # Maybe UTF-16 byte offset (2 bytes per BMP char)
        half_pos = raw_pos // 2
        half_len = max(raw_len // 2, 1)
        if half_pos < text_len:
            end = half_pos + half_len
            if end > text_len:
                half_len = text_len - half_pos
            return half_pos, half_len

        # Give up
        return -1, 0

    def _render_text(self, highlight_pos: int, highlight_len: int):
        """Render self.current_text as HTML with the word at [pos:pos+len] highlighted."""
        try:
            text = self.current_text
            if not text:
                self.display_label.setText("")
                return

            if highlight_pos < 0 or highlight_len <= 0:
                self.display_label.setText(
                    f'<div dir="rtl" style="text-align:right;">'
                    f'{html.escape(text)}</div>'
                )
                return

            # Clamp to valid range
            text_len = len(text)
            if highlight_pos >= text_len:
                self.display_label.setText(
                    f'<div dir="rtl" style="text-align:right;">'
                    f'{html.escape(text)}</div>'
                )
                return
            end = min(highlight_pos + highlight_len, text_len)

            before = text[:highlight_pos]
            word = text[highlight_pos:end]
            after = text[end:]

            highlighted = (
                f'<div dir="rtl" style="text-align:right;">'
                f'{html.escape(before)}'
                f'<span style="background-color: #4FC3F7; color: black; '
                f'border-radius: 3px; padding: 1px 3px;">'
                f'{html.escape(word)}</span>'
                f'{html.escape(after)}'
                f'</div>'
            )
            self.display_label.setText(highlighted)
        except Exception as e:
            print(f"Render error: {e}")
            traceback.print_exc()

    def closeEvent(self, event):
        try:
            if self.speaking:
                self.voice.Speak("", 2)
            pythoncom.CoUninitialize()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TTSHighlightApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
