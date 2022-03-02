"""
General settings and implementation of the SAP Game
"""

from sapai import Player
from sapai import data


class SAP(object):
    score = 0

    def __init__(self):
        self.player = Player(pack="StandardPack") 

    def step(self, action):
        """
        Update the system state every action
        """
        pass #FIXME

    def get_scaled_state(self):
        """
        Get full state, scaled into (approximately) [0, 1].
        State is: 
        team states {id, atk, def, food},
        shop states {id, atk, def},
        money, turn, lives, wins
        """

        team = []
        for teamslot_state in range(self.player.team.state["team"]):
            pet = teamslot_state["pet"]
            team.extend([pet.pet.attack])
        return [self.player.team]


