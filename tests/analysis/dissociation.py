from winak.globaloptimization.disspotter import *

d=DisSpotter('diss.xyz')

a=''
if not d.spot_dis():
    a='not '

print 'Molecule 1 is '+a+'dissociated!'

d=DisSpotter('notdiss.xyz')

a=''
if not d.spot_dis():
    a='not '

print 'Molecule 2 is '+a+'dissociated!'
