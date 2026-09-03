import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def test_home_page_renders_without_exception(self):
        app_path = Path(__file__).with_name("streamlit_app.py")
        app = AppTest.from_file(app_path, default_timeout=10).run()

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "QuantProof")
        self.assertEqual(app.metric[0].label, "Estimated one-way transaction cost")


if __name__ == "__main__":
    unittest.main()
