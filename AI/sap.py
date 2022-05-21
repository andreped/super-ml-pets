"""
General settings and implementation of the SAP Game
"""

from random import randrange, randint
import traceback

import numpy as np

import sapai
from sapai import Player
from sapai import Food
from sapai import Team
from sapai import Pet
from sapai.battle import Battle
from sklearn import random_projection


class SAP(object):
    def __init__(self, data):
        self.player = Player(pack="ExpansionPack1")
        self.score = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.turns = 1
        self.actions_taken_this_turn = 0
        self.past_teams = data.past_teams
        self.logs = data.logs
        self.gen_teams = data.preset_teams

    def step(self, action):
        """
        Update the system state using the best of action (0-68)
        """
        if np.min(action) == 0 and np.max(action) == 0:
            action = len(action) - 1
        else:
            action = np.argmax(action)

        self.actions_taken_this_turn += 1

        if self.actions_taken_this_turn > 20:
            self.score -= 10

        try:
            if action < 35:
                # buyshop
                tm_idx = int(action/7)
                shp_idx = action % 7
                tm_slot = self.player.team[tm_idx]
                shp_slot = self.player.shop[shp_idx]

                if shp_slot.slot_type == "pet":
                    if tm_slot.empty:
                        self.player.buy_pet(shp_slot)
                    else:
                        self.player.buy_combine(shp_slot, tm_slot)
                else:
                    self.player.buy_food(shp_slot, tm_slot)

            elif action < 55:
                # moveteam
                action -= 35
                tm1_idx = int(action/5)
                tm2_idx = action % 5

                if not self.player.team[tm1_idx].empty and self.player.team[tm1_idx].pet.name == self.player.team[tm2_idx].pet.name:
                    self.player.combine(tm2_idx, tm1_idx)
                else:
                    self.player.team.move(tm1_idx, tm2_idx)
            elif action < 60:
                # sellteam
                action -= 55
                tm_slot = self.player.team[action]

                self.player.sell(tm_slot)
            elif action < 67:
                # freezeshop
                action -= 60
                shp_slot = self.player.shop[action]

                self.player.freeze(shp_slot)
            elif action < 68:
                # rollshop
                self.player.roll()
            else:
                # endturn
                self.player.end_turn()

                while len(self.past_teams) <= self.turns:
                    self.past_teams.append([])

                if len(self.past_teams[self.turns]) > 500:
                    self.past_teams[self.turns] = self.past_teams[self.turns][150:]

                enemy = self.generate_enemy()
                battle = Battle(self.player.team, Team(enemy))
                winner = battle.battle()

                if winner == 0:
                    self.wins += 1
                    self.score += 50
                elif winner == 1:
                    self.losses += 1
                    self.score -= 15
                    if self.turns <= 2:
                        self.player.lives -= 1
                    elif self.turns <= 4:
                        self.player.lives -= 2
                    else:
                        self.player.lives -= 3
                else:
                    self.draws += 1
                    self.score += 20

                self.past_teams[self.turns].append(str(list(self.player.team)))
                self.gen_teams.append(str(self.turns) + " " + str(enemy))
                
                self.actions_taken_this_turn = 0
                self.turns += 1

        except Exception:
            self.logs.append(traceback.format_exc())
            
            self.score -= 5

    def clamp(self, val):
        low = 1
        high = min(6, int(self.turns/2) + 1)
        if val < low:
            return low
        if val > high:
            return high

        return val

    def random_pet(self, addl, addh, comp):
        avail_statuses = ["status-honey-bee", "status-bone-attack", "status-garlic-armor",
                          "status-melon-armor", "status-splash-attack", "status-extra-life", "status-steak-attack"]
        tier = min(6, int(self.turns/2) + 1)
        if self.turns <= 3:
            pets = sapai.shop.pet_tier_lookup[tier]
            id = pets[randrange(0, len(pets))]
            spet = sapai.Pet(id)
            spet._attack += randint(addl, addh)
            spet._health += randint(addl, addh)
            spet.experience = randrange(0, self.turns) if tier == 1 else 0
            spet.level = randint(0, 1) if self.turns > 1 else 0
            spet.status = "status-honey-bee" if randint(0, 6) == 0 else "none"

            return spet
        else:
            if comp == 1:  # low tier, high stats
                picktier = self.clamp(randint(tier - 4, tier - 4 + 2))
                pets = sapai.shop.pet_tier_lookup[picktier]
                id = pets[randrange(0, len(pets))]

                if self.turns <= 6:
                    addl = int(1.5*(tier - picktier))
                    addh = addl + 5
                elif self.turns <= 9:
                    addl = int(2*(tier - picktier + 2)) + 7
                    addh = addl + 10
                else: 
                    addl = int(3*self.turns)
                    addh = addh + 15
                spet = sapai.Pet(id)
                spet._attack = min(50, spet._attack + randint(addl, addh))
                spet._health = min(50, spet._health + randint(addl, addh))
                lvl = randrange(0, 5)
                if lvl == 5:
                    exp = 0
                    lvl = 3
                elif lvl >= 2:
                    exp = lvl - 2
                    lvl = 2
                else:
                    exp = lvl
                    lvl = 1
                spet.experience = exp
                spet.level = lvl
                mx = tier if tier != 6 else len(avail_statuses) - 1
                spet.status = avail_statuses[
                    randint(0, mx)] if randint(min(10, self.turns), 10) >= 8 else "none"
            elif comp == 2:  # high tier, low stats
                picktier = self.clamp(randint(tier - 1, tier + 1))
                pets = sapai.shop.pet_tier_lookup[picktier]
                id = pets[randrange(0, len(pets))]

                if self.turns <= 6:
                    addl = int(1.1*(tier - picktier))
                    addh = addl + 3
                elif self.turns <= 10:
                    addl = int(1.7*(tier - picktier)) + 3
                    addh = addl + 4
                else: 
                    addl = int(2*(self.turns))
                    addh = addh + 10
                spet = sapai.Pet(id)
                spet._attack = min(50, spet._attack + randint(addl, addh))
                spet._health = min(50, spet._health + randint(addl, addh))
                lvl = randrange(0, 5)
                if lvl == 5:
                    exp = 0
                    lvl = 3
                elif lvl >= 2:
                    exp = lvl - 2
                    lvl = 2
                else:
                    exp = lvl
                    lvl = 1
                spet.experience = exp
                spet.level = lvl
                mx = tier if tier != 6 else len(avail_statuses) - 1
                spet.status = avail_statuses[
                    randint(0, mx)] if randint(min(10, self.turns), 10) >= 8 else "none"
            else:  # mid
                picktier = self.clamp(randint(tier - 2, tier - 2 + 1))
                pets = sapai.shop.pet_tier_lookup[picktier]
                id = pets[randrange(0, len(pets))]

                if self.turns <= 6:
                    addl = int(1.3*self.turns)
                    addh = addl + 5
                elif self.turns <= 9:
                    addl = int(1.8*self.turns)
                    addh = addl + 6
                else: 
                    addl = int(2.4*self.turns)
                    addh = addh + 13
                spet = sapai.Pet(id)
                spet._attack = min(50, spet._attack + randint(addl, addh))
                spet._health = min(50, spet._health + randint(addl, addh))
                lvl = randrange(0, 5)
                if lvl == 5:
                    exp = 0
                    lvl = 3
                elif lvl >= 2:
                    exp = lvl - 2
                    lvl = 2
                else:
                    exp = lvl
                    lvl = 1
                spet.experience = exp
                spet.level = lvl
                mx = tier if tier != 6 else len(avail_statuses) - 1
                spet.status = avail_statuses[
                    randint(0, mx)] if randint(min(10, self.turns), 10) >= 8 else "none"

            return spet

    def generate_enemy(self):
        team = []
        if self.turns == 1:
            npets = randint(1, 3)
            if npets == 3:
                add = (0, 1)
            elif npets == 2:
                add = (1, 2)
            else:
                add = (2, 3)
        elif self.turns == 2:
            npets = randint(2, 5)
            if npets == 5:
                add = (0, 2)
            elif npets == 4:
                add = (1, 3)
            else:
                add = (2, 3)
        elif self.turns == 3:
            npets = randint(4, 5)
            if npets == 5:
                add = (0, 2)
            else:
                add = (2, 4)
        else:
            npets = 5
            add = (0, 0)

        comp = randint(1, 3)
        for i in range(npets):
            team.append(self.random_pet(add[0], add[1], comp))

        return team

    def get_scaled_state(self):
        """
        Get full state, scaled into (approximately) [0, 1].
        State is: 
        team states {id, exp, atk, def, food},
        shop states {id, atk, def, frozen},
        money, turn, lives, wins
        """
        DATA_LENGTH = len(sapai.data["foods"])+len(sapai.data["pets"])

        state = []
        for teamslot_state in self.player.team.state["team"]:
            pet = teamslot_state["pet"]
            if pet["name"] == "pet-none":
                state.extend([89/len(sapai.data["pets"]), 0, 0, 0, 1])
            else:
                exp = pet["experience"]
                lvl = pet["level"]
                if lvl == 2:
                    exp += 2
                elif lvl == 3:
                    exp = 5

                if pet["status"] != 'none':
                    state.extend([list(sapai.data["pets"].keys()).index(pet["name"])/DATA_LENGTH,
                                  exp/6, pet["attack"]/50, pet["health"]/50,
                                  list(sapai.data["statuses"].keys()).index(pet["status"])/(len(sapai.data["statuses"])+1)])

                else:
                    state.extend([list(sapai.data["pets"].keys()).index(pet["name"])/DATA_LENGTH,
                                  exp/6, pet["attack"]/50, pet["health"]/50, 1])

        for shopslot_state in self.player.shop.state["shop_slots"]:
            item = shopslot_state["item"]
            frozen = 1 if shopslot_state["frozen"] == "frozen" else 0
            if item["name"] == "pet-none" or item["name"] == "food-none":
                state.extend([89/DATA_LENGTH, 0, 0, 0])
            else:
                if item["type"] == "Food":
                    state.extend([(list(sapai.data["foods"].keys()).index(item["name"])+len(sapai.data["pets"])) / DATA_LENGTH,
                                  0, 0, frozen])
                else:
                    state.extend([(list(sapai.data["pets"].keys()).index(item["name"]))/DATA_LENGTH,
                                  item["attack"]/50, item["health"]/50, frozen])

        for i in range(7-len(self.player.shop)):
            state.extend([89/DATA_LENGTH, 0, 0, 0])

        state.extend([self.player.gold, self.player.turn,
                     self.player.lives, self.wins])

        return np.array(state)

    def isGameOver(self):
        if self.player.lives <= 0 or self.wins >= 10 or self.turns >= 30 or self.actions_taken_this_turn > 30:
            return True

        return False
