"""
General settings and implementation of the single-pole cart system dynamics.
"""

from sapai.data import data
from sapai.pets import Pet
from sapai.foods import Food
from sapai.teams import Team,TeamSlot
from sapai.shop import Shop
from sapai.battle import Battle
from sapai import Player


class SAP(object):
    pack = "StandardPack"
    player = Player(pack=pack) 

    def __init__(self):
        pass #FIXME

    def step(self, force):
        """
        Update the system state using leapfrog integration.
            x_{i+1} = x_i + v_i * dt + 0.5 * a_i * dt^2
            v_{i+1} = v_i + 0.5 * (a_i + a_{i+1}) * dt
        """
        pass #FIXME


