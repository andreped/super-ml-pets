from pymeow import *


def read_offsets(proc, base_addr, offsets):
    basepoint = read_int64(proc, base_addr)

    curr_pt = basepoint
    for i in offsets[:-1]:
        curr_pt = read_int64(proc, curr_pt+i)

    return curr_pt + offsets[-1]


process = process_by_name("Super Auto Pets.exe")

base_address = process["modules"]["GameAssembly.dll"]["baseaddr"] + 0x03b1fe20

print("BASE ADDRESS: ", hex(base_address))

obj_offsets = [0x20, 0xb8, 0x40, 0x30, 0x20]

obj_addr = read_offsets(process, base_address, obj_offsets)

print("MAIN CLASS (pointer): ", hex(obj_addr))


turn_addr = read_offsets(process, obj_addr, [0x50])
turn = read_int(process, turn_addr)
print("TURN :", hex(turn_addr), " value:", turn)

money_addr = read_offsets(process, obj_addr, [0x54])
money = read_int(process, money_addr)
print("MONEY :", hex(money_addr), " value:", money)

health_addr = read_offsets(process, obj_addr, [0x40])
health = 10 - read_int(process, health_addr)
print("HEALTH :", hex(health_addr), " value:", health)

wins_addr = read_offsets(process, obj_addr, [0x44])
wins = read_int(process, wins_addr)
print("WINS :", hex(wins_addr), " value:", wins)

print()
print()

minion_dict = {0: "ant", 2: "badger", 3: "beaver", 4: "bee", 5: "bison", 7: "blowfish", 8: "buffalo", 10: "camel", 11: "cat", 12: "caterpillar", 13: "chick", 14: "chicken", 15: "cow", 16: "crab", 17: "cricket", 19: "crocodile", 20: "deer", 21: "dodo", 22: "dog", 23: "dolphin", 25: "dragon", 26: "duck", 27: "eagle", 28: "elephant",
               29: "flamingo", 30: "fly", 32: "fish", 33: "giraffe", 34: "goat", 36: "gorilla", 37: "hedgehog", 38: "hippo", 39: "horse", 40: "kangaroo", 41: "leopard", 44: "llama", 45: "mammoth", 46: "monkey", 47: "mosquito", 50: "octopus", 51: "otter", 52: "ox", 53: "parrot", 54: "peacock", 55: "rhino", 56: "penguin", 57: "rat", 59: "pig", 60: "rabbit", 62: "ram", 63: "rooster", 65: "scorpion", 66: "seal", 67: "shrimp", 68: "sheep", 69: "shark", 70: "skunk", 71: "sloth", 72: "snail", 73: "snake", 74: "spider", 75: "squirrel", 76: "swan", 77: "tiger", 78: "tyrannosaurus", 79: "turkey", 80: "turtle", 81: "whale", 82: "worm", 85: "bus", 86: "zombie cricket", 87: "bat", 88: "beetle", 89: "bluebird", 91: "hatching chick", 92: "ladybug", 93: "lobster", 94: "microbe", 95: "owl", 96: "poodle", 97: "puppy", 98: "sauropod", 99: "tabby cat", 100: "tropical fish", 103: "boar", 104: "dromedary"}
print(len(minion_dict), "minions identified")
minion_shop_list_addr = read_offsets(process, obj_addr, [0x78, 0x10])
print("SHOP MINIONS :", hex(minion_shop_list_addr))
minion_arr_offsets = [0x20, 0x28, 0x30,
                        0x38, 0x40, 0x48, 0x50]  # , 0x58, 0x60, 0x68
for i in range(len(minion_arr_offsets)):
    try:
        minion = read_offsets(process, minion_shop_list_addr, [
                              minion_arr_offsets[i]])
        minion_id_addr = read_offsets(process, minion, [0x34])
        minion_atk_addr = read_offsets(process, minion, [0x58, 0x10])
        minion_def_addr = read_offsets(process, minion, [0x50, 0x10])
        minion_frozen_addr = read_offsets(process, minion, [0x28])

        minion_id = read_int(process, minion_id_addr)
        minion_name = minion_dict[minion_id] if minion_id in minion_dict else "unknown"
        minion_atk = read_int(process, minion_atk_addr)
        minion_def = read_int(process, minion_def_addr)
        minion_frozen = read_int(process, minion_frozen_addr) == 1

        print("Minion", i, "ID :", hex(minion_id_addr),
              " value: ", minion_id, " (" + minion_name + ")")
        print("Minion", i, "atk :", hex(minion_atk_addr),
              " value: ", minion_atk)
        print("Minion", i, "def :", hex(minion_def_addr),
              " value: ", minion_def)
        print("Minion", i, "frozen :", hex(minion_frozen_addr),
              " value: ", minion_frozen)
        print()
    except:
        print("empty slot")

print()
print()


# food shop listed backwards
food_dict = {0: "apple", 9: "meat bone", 16: "canned food", 22: "chili", 23: "chocolate", 38: "garlic", 40: "honey", 49: "milk", 
             50: "cupcake", 51: "mushroom", 58: "pear", 63: "pizza", 73: "salad bowl", 79: "steak", 82: "sushi", 92: "sleeping pill", 96: "melon"}
print(len(food_dict), " foods identified")
food_shop_list_addr = read_offsets(process, obj_addr, [0x88, 0x10])
print("SHOP FOOD :", hex(food_shop_list_addr))
food_arr_offsets = [0x20, 0x28, 0x30, 0x38, 0x40, 0x48, 0x50] #, 0x58, 0x60, 0x68
for i in range(len(food_arr_offsets)):
    try:
        food = read_offsets(process, food_shop_list_addr,
                            [food_arr_offsets[i]])
        food_id_addr = read_offsets(process, food, [0x30])
        food_frozen_addr = read_offsets(process, food, [0x28])

        food_id = read_int(process, food_id_addr)
        food_name = food_dict[food_id] if food_id in food_dict else "unknown"
        food_frozen = read_int(process, food_frozen_addr) == 1

        print("Food", i, "ID :", hex(food_id_addr),
                " value: ", food_id, " (" + food_name + ")")
        print("Food", i, "frozen :", hex(food_frozen_addr),
                " value: ", food_frozen)
        print()
    except:
        print("empty slot")

print()
print()

perk_dict = {0: "none", 1: "status-extra-life", 2: "status-poision-attack", 5: "status-splash-attack", 6: "status-bone-attack",
             8: "status-honey-bee", 9: "status-garlic-armor", 10: "status-weak", 12: "status-steak-attack", 13: "status-melon-armor"}
print(len(perk_dict), "perks identified")
minion_team_list_addr = read_offsets(process, obj_addr, [0x60, 0x18, 0x10])
print("My Minions: ", hex(minion_team_list_addr))
minion_arr_offsets = [0x20, 0x28, 0x30,
                        0x38, 0x40]  # , 0x48, 0x50, 0x58, 0x60, 0x68
for i in range(len(minion_arr_offsets)):
    try: 
        minion = read_offsets(process, minion_team_list_addr, [
                                minion_arr_offsets[i]])
        minion_id_addr = read_offsets(process, minion, [0x34])
        minion_atk_addr = read_offsets(process, minion, [0x58, 0x10])
        minion_def_addr = read_offsets(process, minion, [0x50, 0x10])
        minion_atk_tmp_addr = read_offsets(process, minion, [0x58, 0x14])
        minion_def_tmp_addr = read_offsets(process, minion, [0x50, 0x14])
        minion_exp_addr = read_offsets(process, minion, [0x44])
        minion_perk_addr = read_offsets(process, minion, [0x64])

        minion_id = read_int(process, minion_id_addr)
        minion_name = minion_dict[minion_id] if minion_id in minion_dict else "unknown"
        minion_atk = read_int(process, minion_atk_addr)+read_int(process, minion_atk_tmp_addr)
        minion_def = read_int(process, minion_def_addr)+read_int(process, minion_def_tmp_addr)
        minion_exp = read_int(process, minion_exp_addr)
        minion_perk_id = read_int(process, minion_perk_addr)
        minion_perk = perk_dict[minion_perk_id] if minion_perk_id in perk_dict else "unknown"

        print("Minion", i, "ID :", hex(minion_id_addr),
                " value: ", minion_id, " (" + minion_name + ")")
        print("Minion", i, "atk :", hex(minion_atk_addr),
                " value: ", minion_atk)
        print("Minion", i, "def :", hex(minion_def_addr),
                " value: ", minion_def)
        print("Minion", i, "exp :", hex(minion_exp_addr),
              " value: ", minion_exp)
        print("Minion", i, "perk :", hex(minion_perk_addr),
                " value: ", minion_perk_id, " (" + minion_perk + ")")
        print()
    except:
        print("empty slot")
