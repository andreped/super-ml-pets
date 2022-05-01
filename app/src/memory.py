from pymeow import *


def read_offsets(proc, base_addr, offsets):
    basepoint = read_int64(proc, base_addr)

    curr_pt = basepoint
    for i in offsets[:-1]:
        curr_pt = read_int64(proc, curr_pt+i)

    return curr_pt + offsets[-1]


process = process_by_name("Super Auto Pets.exe")

base_address = process["modules"]["GameAssembly.dll"]["baseaddr"] + 0x029A8598

print("BASE ADDRESS: ", hex(base_address))

obj_offsets = [0x80, 0x50, 0xb8, 0x20, 0x30, 0x20]

obj_addr = read_offsets(process, base_address, obj_offsets)

print("MAIN CLASS (pointer): ", hex(obj_addr))


turn_addr = read_offsets(process, obj_addr, [0x50])
print("TURN :", hex(turn_addr), " value:", read_int(process, turn_addr))

money_addr = read_offsets(process, obj_addr, [0x54])
print("MONEY :", hex(money_addr), " value:", read_int(process, money_addr))

health_addr = read_offsets(process, obj_addr, [0x40])
print("HEALTH :", hex(health_addr), " value:",
      10 - read_int(process, health_addr))

wins_addr = read_offsets(process, obj_addr, [0x44])
print("WINS :", hex(wins_addr), " value:", read_int(process, wins_addr))

print()
print()

minion_dict = {0: "ant", 3: "beaver", 7:"blowfish", 10: "camel", 16: "crab", 17: "cricket", 21: "dodo", 22: "dog", 26: "duck", 28: "elephant",
               29: "flamingo", 32: "fish", 33: "giraffe", 37: "hedgehog", 39: "horse",40:"kangaroo", 47: "mosquito", 51: "otter", 52: "ox", 54: "peacock", 57: "rat", 59: "pig",60:"rabbit", 67: "shrimp", 68: "sheep", 72:"snail", 74: "spider", 76: "swan", 80:"turtle"}
print(len(minion_dict), "minions identified")
minion_shop_list_addr = read_offsets(process, obj_addr, [0x78, 0x10])
print("SHOP MINIONS :", hex(minion_shop_list_addr))
try:
    minion_arr_offsets = [0x20, 0x28, 0x30,
                          0x38, 0x40, 0x48, 0x50, 0x58, 0x60, 0x68]
    for i in range(10):
        minion = read_offsets(process, minion_shop_list_addr, [minion_arr_offsets[i]])
        minion_id = read_offsets(process, minion, [0x34])
        minion_atk = read_offsets(process, minion, [0x58, 0x10])
        minion_def = read_offsets(process, minion, [0x50, 0x10])
        minion_frozen = read_offsets(process, minion, [0x28])
        print("Minion", i, "ID :", hex(minion_id),
              " value: ", read_int(process, minion_id))
        print("Minion", i, "atk :", hex(minion_atk),
              " value: ", read_int(process, minion_atk))
        print("Minion", i, "def :", hex(minion_def),
              " value: ", read_int(process, minion_def))
        print("Minion", i, "frozen :", hex(minion_frozen),
              " value: ", read_int(process, minion_frozen)==1)
        print()
except:
    print("no more minions in shop")

print()
print()


# food shop listed backwards
food_dict = {0: "apple", 9: "meat bone", 38: "garlic", 40: "honey", 50: "cupcake", 73: "salad bowl", 92: "sleeping pill"}
print(len(food_dict), " foods identified")
food_shop_list_addr = read_offsets(process, obj_addr, [0x88, 0x10])
print("SHOP FOOD :", hex(food_shop_list_addr))
try:
    food_arr_offsets = [0x20, 0x28, 0x30,
                            0x38, 0x40, 0x48, 0x50, 0x58, 0x60, 0x68]
    for i in range(10):
        food = read_offsets(process, food_shop_list_addr, [food_arr_offsets[i]])
        food_id = read_offsets(process, food, [0x30])
        food_frozen = read_offsets(process, food, [0x28])
        print("Food", i, "ID :", hex(food_id),
                " value: ", read_int(process, food_id))
        print("Food", i, "frozen :", hex(food_frozen),
                " value: ", read_int(process, food_frozen)==1)
        print()
except:
    print("no more foods in shop")

print()
print()

perk_dict = {0: "none", 6: "status-bone-attack", 8: "status-honey-bee", 9: "status-garlic-armor"}
print(len(perk_dict), "perks identified")
minion_team_list_addr = read_offsets(process, obj_addr, [0x60, 0x18, 0x10])
print("My Minions: ", hex(minion_team_list_addr))
try:
    minion_arr_offsets = [0x20, 0x28, 0x30,
                            0x38, 0x40, 0x48, 0x50, 0x58, 0x60, 0x68]
    for i in range(10):
        minion = read_offsets(process, minion_team_list_addr, [minion_arr_offsets[i]])
        minion_id = read_offsets(process, minion, [0x34])
        minion_atk = read_offsets(process, minion, [0x58, 0x10])
        minion_def = read_offsets(process, minion, [0x50, 0x10])
        minion_atk_tmp = read_offsets(process, minion, [0x58, 0x14])
        minion_def_tmp = read_offsets(process, minion, [0x50, 0x14])
        minion_perk = read_int64(process, minion) + 0x60
        print("Minion", i, "ID :", hex(minion_id),
                " value: ", read_int(process, minion_id))
        print("Minion", i, "atk :", hex(minion_atk),
                " value: ", read_int(process, minion_atk)+read_int(process, minion_atk_tmp))
        print("Minion", i, "def :", hex(minion_def),
                " value: ", read_int(process, minion_def)+read_int(process, minion_def_tmp))
        print("Minion", i, "perk :", hex(minion_perk),
                " value: ", read_int(process, minion_perk))
        print()
except:
    print("no more minions in team")
