import pickle

e_file = open('/home/pi/Zlawren1_Swimbot/Epsilon/2023_04_23_12_17_07_episode:19.file', 'rb')
e = pickle.load(e_file)
e_file.close()
print(e)