from smp.actions import *


def test_class():
    sap = SuperAutoPetsMouse()

    sap.end_turn(1)
    sap.freeze_unfreeze(1)
    sap.roll()
    sap.get_action_dict()
    sap.reorder(order=[list(range(5))])
    #sap.combine_in_team(n=[1, 1])