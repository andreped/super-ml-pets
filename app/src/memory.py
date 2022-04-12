from ReadWriteMemory import ReadWriteMemory

rwm = ReadWriteMemory()
Get a Process by name
from ReadWriteMemory import ReadWriteMemory

rwm = ReadWriteMemory()

process = rwm.get_process_by_name('ac_client.exe')
Get a Process by ID
from ReadWriteMemory import ReadWriteMemory

rwm = ReadWriteMemory()

process = rwm.get_process_by_id(1337)
Get the list of running processes ID's from the current system
from ReadWriteMemory import ReadWriteMemory

rwm = ReadWriteMemory()

processes_ids = rwm.enumerate_processes()
Print the Process information
from ReadWriteMemory import ReadWriteMemory

rwm = ReadWriteMemory()

process = rwm.get_process_by_name('ac_client.exe')
print(process.__dict__)
Print the Process HELP docs
from ReadWriteMemory import ReadWriteMemory

rwm = ReadWriteMemory()

process = rwm.get_process_by_name('ac_client.exe')
help(process)
Exception: ReadWriteMemoryError
from ReadWriteMemory import ReadWriteMemory
from ReadWriteMemory import ReadWriteMemoryError

rwm = ReadWriteMemory()
try:
    process = rwm.get_process_by_name('ac_client.exe')
except ReadWriteMemoryError as error:
    print(error)
Open the Process
To be able to read or write to the process's memory first you need to call the open() method.

from ReadWriteMemory import ReadWriteMemory

rwm = ReadWriteMemory()

process = rwm.get_process_by_name('ac_client.exe')
process.open()
Set the pointers for example: to get health, ammo and grenades
The offsets must be a list in the correct order, if the address does not have any offsets then just pass the address. You need to pass two arguments, first the process address as hex and a list of offsets as hex.

from ReadWriteMemory import ReadWriteMemory

rwm = ReadWriteMemory()

process = rwm.get_process_by_name('ac_client.exe')
process.open()

health_pointer = process.get_pointer(0x004e4dbc, offsets=[0xf4])
ammo_pointer = process.get_pointer(0x004df73c, offsets=[0x378, 0x14, 0x0])
grenade_pointer = process.get_pointer(0x004df73c, offsets=[0x35c, 0x14, 0x0])
Read the values for the health, ammo and grenades from the Process's memory
from ReadWriteMemory import ReadWriteMemory

rwm = ReadWriteMemory()

process = rwm.get_process_by_name('ac_client.exe')
process.open()

health_pointer = process.get_pointer(0x004e4dbc, offsets=[0xf4])
ammo_pointer = process.get_pointer(0x004df73c, offsets=[0x378, 0x14, 0x0])
grenade_pointer = process.get_pointer(0x004df73c, offsets=[0x35c, 0x14, 0x0])

health = process.read(health_pointer)
ammo = process.read(ammo_pointer)
grenade = process.read(grenade_pointer)
Print the health, ammo and grenade values
from ReadWriteMemory import ReadWriteMemory

rwm = ReadWriteMemory()

process = rwm.get_process_by_name('ac_client.exe')
process.open()

health_pointer = process.get_pointer(0x004e4dbc, offsets=[0xf4])
ammo_pointer = process.get_pointer(0x004df73c, offsets=[0x378, 0x14, 0x0])
grenade_pointer = process.get_pointer(0x004df73c, offsets=[0x35c, 0x14, 0x0])

health = process.read(health_pointer)
ammo = process.read(ammo_pointer)
grenade = process.read(grenade_pointer)

print({'Health': health, 'Ammo': ammo, 'Grenade': grenade})
Write some random values for health, ammo and grenade to the Process's memory
from ReadWriteMemory import ReadWriteMemory
from random import randint

rwm = ReadWriteMemory()

process = rwm.get_process_by_name('ac_client.exe')
process.open()

health_pointer = process.get_pointer(0x004e4dbc, offsets=[0xf4])
ammo_pointer = process.get_pointer(0x004df73c, offsets=[0x378, 0x14, 0x0])
grenade_pointer = process.get_pointer(0x004df73c, offsets=[0x35c, 0x14, 0x0])

process.write(health_pointer, randint(1, 100))
process.write(ammo_pointer, randint(1, 20))
process.write(grenade_pointer, randint(1, 5))
Close the Process's handle when you are done using it.
from ReadWriteMemory import ReadWriteMemory

rwm = ReadWriteMemory()

process = rwm.get_process_by_name('ac_client.exe')
process.open()

process.close()
Examples
Check out the code inside the Test folder on the python file named testing_script.py. The AssaultCube game used for this test is version v1.1.0.4 If you use a different version then you will have to use CheatEngine to find the memory addresses. https://github.com/assaultcube/AC/releases/tag/v1.1.0.4
For more examples check out the AssaultCube game trainer: https://github.com/vsantiago113/ACTrainer

from ReadWriteMemory import ReadWriteMemory
from random import randint

rwm = ReadWriteMemory()
process = rwm.get_process_by_name('ac_client.exe')
process.open()

print('\nPrint the Process information.')
print(process.__dict__)

health_pointer = process.get_pointer(0x004e4dbc, offsets=[0xf4])
ammo_pointer = process.get_pointer(0x004df73c, offsets=[0x378, 0x14, 0x0])
grenade_pointer = process.get_pointer(0x004df73c, offsets=[0x35c, 0x14, 0x0])
print(health_pointer)

health = process.read(health_pointer)
ammo = process.read(ammo_pointer)
grenade = process.read(grenade_pointer)

print('\nPrinting the current values.')
print({'Health': health, 'Ammo': ammo, 'Grenade': grenade})

process.write(health_pointer, randint(1, 100))
process.write(ammo_pointer, randint(1, 20))
process.write(grenade_pointer, randint(1, 5))

health = process.read(health_pointer)
ammo = process.read(ammo_pointer)
grenade = process.read(grenade_pointer)

print('\nPrinting the new modified random values.')
print({'Health': health, 'Ammo': ammo, 'Grenade': grenade})

process.close()