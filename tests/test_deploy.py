from smp.deploy_agent import remove_nothing, get_action_name, time_pause
from smp.image_detection import find_paw, find_arena, get_image_directory, find_the_animals
import os


def test_remove_nothing():
    remove_nothing(["1", "2", "nothing", "3"])


def test_get_action_name():
    get_action_name(0)


def test_pause():
    time_pause(0)


def test_find_paw():
    find_paw()


def test_find_arena():
    find_arena()


def test_find_animals():
    find_the_animals(
        directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../smp/SAP_res/").replace("\\", "/")
    )
