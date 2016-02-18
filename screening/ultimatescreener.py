import numpy as np
from datetime import datetime
from ase.io.trajectory import Trajectory
from collections import Counter
from winak.screening.composition import Stoichiometry
from ase.utils.geometry import sort
import os 


class UltimateScreener:
    """UltimateScreener

    by Konstantin Krautgasser, November 2015
    """

    def __init__(self, atoms,
                 EnergyEvaluator,
                 Displacer,
                 Criterion,
                 trajectory='minima.traj',
                 logfile='tt.log'):
        self.atoms=atoms
        self.logfile=logfile
        self.eneval=EnergyEvaluator
        self.displacer=Displacer
        self.crit=Criterion
        self.traj=Trajectory(trajectory,'w')
        self.startT = datetime.now()
        self.log('STARTING Screening at '+self.startT.strftime('%Y-%m-%d %H:%M:%S')+' KK 2015')
        self.log('Using the following Parameters and Classes:')
        self.log('EnergyEvaluator - '+self.eneval.print_params())
        self.log('Displacer - '+self.displacer.print_params())
        self.log('Criterion - '+self.crit.print_params())


    def run(self, steps):
        """Screen for defined number of steps."""
        tmp = self.eneval.get_energy(self.atoms.copy())
        if tmp is None:
            self.log('Initial Energy Evaluation Failed. Please check your calculator!')
        else:
            self.current=tmp[0];self.Emin=tmp[1]
            self.traj.write(self.current)
            self.log('Initial Energy Evaluation done. Note that this is structure 0 in your trajectory.')
        self.fallbackatoms=self.current.copy()
        self.fallbackstep=-1

        for step in range(steps):    
            """I strictly use copies here, so nothing can be overwritten in a subclass.
            A step is tried 10 times, if it keeps failing, the structure is reset.
            If it still fails, abort, since something is clearly wrong."""
            tmp=None
            tries=0
            reset=False  
            failed=False          
            
            while tmp is None:
                try:
                    tmp=self.displacer.displace(self.current.copy())
                except:
                    tmp=None
                tries+=1
                if tmp is None:
                    self.log('Error while displacing, retrying')
                else:
                    tmp=self.eneval.get_energy(tmp.copy())
                    if tmp is None:
                        #os.system('cp dftb.out '+str(step+1).zfill(2)+'_'+str(tries).zfill(2)+'.fail')
                        self.log('Error while evaluating energy, retrying')
                if tries>10 and not reset:
                    self.log('Repeated Error during current step, rolling back to step %d' %self.fallbackstep)
                    #self.fallbackatoms.write('eee.xyz')
                    self.current=self.fallbackatoms.copy()
                    #self.current.write('fff.xyz')
                    tries=0
                    reset=True
                if tries>10 and reset:
                    failed=True
                    break
            if failed:
                self.log('ABORTING. COULD NOT PERFORM STEP.')
                break
            
            """change trajectory if stoichiometry has changed"""
            comp=Stoichiometry()
            oldtraj=self.traj
            if comp.has_changed(tmp[0],self.current):
                newtraj = comp.make_traj()
                self.traj=newtraj
            
            """setting backup"""
            self.traj.write(tmp[0])
            accepted=self.crit.evaluate(tmp[0].copy(),tmp[1])
            acc='not'
            if accepted:       
                self.current=tmp[0].copy()
                acc=''
                self.fallbackatoms=tmp[0].copy()
                self.fallbackstep=step
            else:
                self.traj=oldtraj
            #self.log('%s - step %d done, %s accepted, Energy = %f '%(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),step+1,acc,tmp[1]))
            self.log('%s - step %d done, %s accepted, Energy = %f, Stoichiometry = %s '%(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),step+1,acc,tmp[1], comp.stoich))
        self.endT = datetime.now()
        self.log('ENDING Screening at '+self.endT.strftime('%Y-%m-%d %H:%M:%S'))
        self.log('Time elapsed: '+str(self.endT-self.startT))
            
       
    def log(self, msg):
        if self.logfile is not None:
            with open(self.logfile,'a') as fn:
                fn.write(msg+'\n')
                fn.flush()
