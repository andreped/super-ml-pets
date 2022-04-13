from ReadWriteMemory import ReadWriteMemory
from random import randint

rwm = ReadWriteMemory()

process = rwm.get_process_by_name('ac_client.exe')

print(process.__dict__)
help(process)

process.open()

# Set the pointers for example: to get health, ammo and grenades
# The offsets must be a list in the correct order, if the address does not have any offsets then just pass the address. You need to pass two arguments, first the process address as hex and a list of offsets as hex.


health_pointer = process.get_pointer(0x004e4dbc, offsets=[0xf4])
ammo_pointer = process.get_pointer(0x004df73c, offsets=[0x378, 0x14, 0x0])
grenade_pointer = process.get_pointer(0x004df73c, offsets=[0x35c, 0x14, 0x0])

health = process.read(health_pointer)
ammo = process.read(ammo_pointer)
grenade = process.read(grenade_pointer)

print({'Health': health, 'Ammo': ammo, 'Grenade': grenade})


process.write(health_pointer, randint(1, 100))
process.write(ammo_pointer, randint(1, 20))
process.write(grenade_pointer, randint(1, 5))


process.close()
