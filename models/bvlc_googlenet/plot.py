#import pickle
acc = []
with open("copy.log", "r") as f:
  for line in f:
    if "Testing net (#0)" in line:
        print line[0:-1]
    if "Test net output #" in line:
        print line[0:-1]
        ss = line.split(" ")
        acc.append(ss[-1][0:-1])

print max(acc)

'''
loss_value = []
with open("recurrent_copy.log", "r") as f:
  for line in f:
    if ", loss = " in line:
        ss = line.split(" ")
        print ss[-1]
        loss_value.append(ss[-1][0:-1])
'''

#pickle.dump( loss_value, open( "loss_value", "wb" ) )
