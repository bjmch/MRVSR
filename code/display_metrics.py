import os

net = 'mrvsr'
cmd = 'python aggregate_metrics.py --data QSV --net ' + net + ' --metric psnr'
os.system(cmd)
cmd = 'python aggregate_metrics.py --data QSV --net ' + net + ' --metric ssim'
os.system(cmd)

net = 'rfs3'
cmd = 'python aggregate_metrics.py --data QSV --net ' + net + ' --metric psnr'
os.system(cmd)
cmd = 'python aggregate_metrics.py --data QSV --net ' + net + ' --metric ssim'
os.system(cmd)

net = 'rlsp'
cmd = 'python aggregate_metrics.py --data QSV --net ' + net + ' --metric psnr'
os.system(cmd)
cmd = 'python aggregate_metrics.py --data QSV --net ' + net + ' --metric ssim'
os.system(cmd)

net = 'mrvsr'
cmd = 'python aggregate_metrics.py --data Vid4 --net ' + net + ' --metric psnr'
os.system(cmd)
net = 'rfs3'
cmd = 'python aggregate_metrics.py --data Vid4 --net ' + net + ' --metric psnr'
os.system(cmd)
