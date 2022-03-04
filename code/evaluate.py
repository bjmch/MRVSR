import os

net = 'mrvsr'
for sequence in [1, 2, 3, 4]:
    cmd = 'python run_network.py --data QSV --net ' + net + ' --sequence ' + str(sequence)
    os.system(cmd)

net = 'rfs3'
for sequence in [1, 2, 3, 4]:
    cmd = 'python run_network.py --data QSV --net ' + net + ' --sequence ' + str(sequence)
    os.system(cmd)

net = 'rlsp'
for sequence in [1, 2, 3, 4]:
    cmd = 'python run_network.py --data QSV --net ' + net + ' --sequence ' + str(sequence)
    os.system(cmd)

net = 'mrvsr'
for sequence in ['calendar', 'city', 'foliage', 'walk']:
    cmd = 'python run_network.py --data Vid4 --net ' + net + ' --sequence ' + str(sequence)
    os.system(cmd)

net = 'rfs3'
for sequence in ['calendar', 'city', 'foliage', 'walk']:
    cmd = 'python run_network.py --data Vid4 --net ' + net + ' --sequence ' + str(sequence)
    os.system(cmd)
