from smp.utils import get_position
import os


def test_position():
    os.environ['DISPLAY'] = ':0'
    get_position()
    