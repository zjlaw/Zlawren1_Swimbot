import pickle

e_file = open('Epsilon/2023_04_22_10_17_32_episode:0.file', 'rb')
e = pickle.load(e_file)
e_file.close()
print(e)