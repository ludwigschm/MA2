"""Kompatibilitätsmodul: nutzt die Implementierung aus app_kivy2."""

from app_kivy2 import TouchGameApp  # noqa: F401


if __name__ == "__main__":
    TouchGameApp().run()
