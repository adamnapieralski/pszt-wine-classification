import os
import shutil
import urllib

# create data directory
data_dir_path = 'data/'
print("Creating " + data_dir_path + " directory (if not exists)...")
try: 
    os.makedirs(data_dir_path)
except OSError:
    if not os.path.isdir(data_dir_path):
        raise
        
# clear its contents
print("Removing " + data_dir_path + " previous contents...")
for filename in os.listdir(data_dir_path):
    file_path = os.path.join(data_dir_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete {}. Reason: {}'.format(file_path, e))

# download data files
data_urls= []
with open('data_urls.txt') as f:
    data_urls = f.readlines()
    
for url in data_urls:
    print("Downloading " + data_dir_path + os.path.basename(url).rstrip() + "...")
    datafile = urllib.URLopener()
    datafile.retrieve(url, data_dir_path + os.path.basename(url))
    print("Finished downloading.")

