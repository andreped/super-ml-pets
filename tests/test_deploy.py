from smp.deploy_agent import remove_nothing, get_action_name, time_pause
from smp.image_detection import find_paw, find_arena, get_image_directory, find_the_animals
from main import main
import os, sys
import subprocess as sp


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


def test_deploy(monkeypatch):
    try:
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['main', '--task', 'deploy', '--infer_model',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model_sap_gym_sb3_210822_finetuned_212756_steps").replace("\\", "/")]
            )

            main()
    except Exception as e:
        print(e)
        pass
