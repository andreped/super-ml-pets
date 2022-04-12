import glob, os


list_of_files = glob.glob('ckpt/*')
latest_file = max(list_of_files, key=os.path.getctime)

print(latest_file.name)