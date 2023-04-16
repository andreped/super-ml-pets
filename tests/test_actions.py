from smp.actions import *


def test_class():
    sap = SuperAutoPetsMouse()

    sap.end_turn(1)
    sap.freeze_unfreeze(1)
    sap.roll()
    sap.get_action_dict()
    sap.reorder(order=[list(range(5))])
    #sap.combine_in_team(n=[1, 1])
    for i in range(5):
        sap.buy(nth_slot=[i])

    for j in [5, 6]:
        sap.buy_food(nth_slot=[j], num_pets=5)
    
    sap.sell(nth_team_slot=[4])
    sap.combine_in_team(n=[1, 2])
    sap.buy_combine(n=[1, 2])
    sap.freeze_unfreeze(nth_slot=2)

    sap.roll()
    sap.end_turn(0)

