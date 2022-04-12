import pickle
import csv
import traceback

from sapai import Team, Pet

try:
    with open('past_teams', 'r') as f:
        ls = [[]]
        turn = 0
        while True:
            a = f.readline()
            if not a:
                break
            if a == "\n":
                ls.append([])
                turn += 1
                continue
            
            t = []
            a = a.split(" ")
            if a[3] == "EMPTY":
                p = Pet("pet-none")
            else:
                p = Pet(a[3])

                st = a[4].split("-")
                p._attack = int(st[0])
                p._health = int(st[1])
                p.status = a[5]
                p.level = int(a[6][0])
                p.experience = int(a[6][2])
            
            t.append(p)
            for i in range(4):
                a = f.readline()        
                a = a.split(" ")
                if a[3] == "EMPTY":
                    p = Pet("pet-none")
                else:
                    p = Pet(a[3])

                    st = a[4].split("-")
                    p._attack = int(st[0])
                    p._health = int(st[1])
                    p.status = a[5]
                    p.level = int(a[6][0])
                    p.experience = int(a[6][2])
                
                t.append(p)

            ls[turn].append(Team([]))

        with open('past_teams_bin', 'wb') as f:
            pickle.dump(ls, f)

except Exception:
    print(a)
    traceback.print_exc()