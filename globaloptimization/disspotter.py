from ase.all import *
from INTERNALS.curvilinear.InternalCoordinates import icSystem
from INTERNALS.curvilinear.InternalCoordinates import ValenceCoordinateGenerator as VCG
from INTERNALS.curvilinear.Coordinates import InternalCoordinates as IC


class DisSpotter:
    """
    Tells you if a molecule is dissociated or not.
    """
    def __init__(self, atoms):
        if isinstance(atoms,basestring):
            self.molecule=read(file)
        else:
            self.molecule=atoms
        self.visited=[False]*(len(self.molecule)+1)
        self.vcg=VCG(self.molecule.get_chemical_symbols(),masses=self.molecule.get_masses())
        self.iclist=self.vcg(self.molecule.get_positions().flatten())
        self.ic=icSystem(self.iclist,len(self.molecule),masses=self.molecule.get_masses(),xyz=self.molecule.get_positions().flatten())

        self.stre=self.ic.getStretchBendTorsOop()[0][0]
        self.ics=self.ic.getStretchBendTorsOop()[1]

    def numConnections(self,j):
        """ics: as in the main program bottom
           stre: list of stretches
           j: atom you want to find
        """
        found=0
        s=[]
        for i in self.stre:
            a=self.ics[i]
            if a[0]==j or a[1]==j:
                found+=1
                s.append(i)
        return found,s

    def fragment(self,j):
        """
        This might seem complicated, but it is actually not.
        """
        ret=0
        if not self.visited[j]:
            ret+=1
            self.visited[j]=True
            x=self.numConnections(j)
            #print 'looking at '+str(j)
            #print str(j)+' has '+str(x[0])+' connections'
            if x[0]==1:
                #print 'ending in atom '+str(j)+' and returning '+str(ret)
                return ret
            else:
                for l in x[1]:
                    tmp=self.ics[l][1]
                    if tmp==j:
                        tmp=self.ics[l][0]
                    #print str(j)+' is connected to atom '+str(tmp)
                    ret+=self.fragment(tmp)
                    #print 'back at atom '+str(j)
                    #print 'ret is now '+str(ret)
            return ret
        else:
            #print 'Atom '+str(j)+' already accounted for'
            return 0

    def spot_dis(self):
        """
        Returns true if dissociation happened
        """
        return len(self.molecule)>self.fragment(1)#1 is arbitrary