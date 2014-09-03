from INTERNALS.globaloptimization.disspotter import *

d=DisSpotter(file='diss.xyz')

a=''
if not d.spot_dis():
    a='not '

print 'Molecule 1 is '+a+'dissociated!'

d=DisSpotter(file='notdiss.xyz')

a=''
if not d.spot_dis():
    a='not '

print 'Molecule 2 is '+a+'dissociated!'