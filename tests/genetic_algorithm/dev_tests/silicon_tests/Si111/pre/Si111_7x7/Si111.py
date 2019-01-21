from ase.all import *
from ase.lattice.surface import *
from ase.data import *
import numpy as np

tol=0.1
huc_line=( (13.51281059,23.40487449) , (27.02562118,0.0) )
huc_vert=(40.538,23.40487449)

def centro(v1,v2,v3): #baricentro triangolo con vertici v1 v2 v3
    return ( (v1[0]+v2[0]+v3[0])/3 , (v1[1]+v2[1]+v3[1])/3 )

### SATURATION = replace bottom layer with H
def saturate(slab):
    bottom_layer = [atom for atom in slab if atom.tag==nz+1]
    for atom in bottom_layer:
        atom.symbol='H'
        atom.z += 0.8

### CONSTRAINTS = constrain the first n layers (excluding saturation layer)
def constrain_bottom_layers(slab, n):
    #n = input('input number of layers to constrain: ')
    mask = [atom.tag > nz-n for atom in slab]
    c = FixAtoms(mask=mask)
    slab.set_constraint(c)

def layer(slab,n):
    return [atom for atom in slab if atom.tag==n]

def color_layers(slab,restore=False):   # change atomic type of first three layer to facilitate reconstuction, restore=True restores all Si
    for nlayer in range(1,5):
        layer = [atom for atom in slab if atom.tag==nlayer]
        for atom in layer:
            atom.symbol=chemical_symbols[14+nlayer]
    if restore:
        for atom in [atom for atom in slab if not atom.symbol=='H']:
            atom.symbol='Si'

def del_layer(slab,n):
    del slab[[atom.index for atom in slab if atom.tag==n]]
    #reassign tags below
    for i in range(n,nz+1):
        for atom in layer(slab,i):
            atom.tag = i-1

def del_corner_atom(slab):
    dy = [atom.y for atom in slab if atom.tag==2 and atom.x == 0.0]#and abs(atom.x-6*lc/sqrt(2))<0.1]]# == 0.0]]
    del slab[[atom.index for atom in slab if atom.tag==2 and atom.x == 0.0]]#and abs(atom.x-6*lc/sqrt(2))<0.1]]# == 0.0]]
    slab.translate((0,-dy[0],0))

def dimerize(slab): ### DA RIFARE CON DET
    L = 7*dl
    dx=0.5
    for i in range(1,7):
        even = i%2==0
        ## orizz    (x,y) = (l,0)    
        ## asc      (x,y) = (l/2, l*sqrt(3)/2)
        ## disc     (x,y)=  (L-l/2, l*sqrt(3)/2)
        ## l=(atom.x*sqrt(2)/lc)
        for atom in layer(slab,2):
            if abs(atom.y-0.0)<tol and abs(atom.x-i*dl)<tol:
                atom.symbol='F'
                if even:
                    atom.x -= dx
                else:
                    atom.x += dx
            elif abs(atom.x-i*dl/2)<tol and abs(atom.y-i*dl*sqrt(3)/2)<tol: #and atom.y==dl*sqrt(3)/2:
                atom.symbol='F'
                if even:
                    atom.x -= dx/2; atom.y -= dx*sqrt(3)/2
                else:
                    atom.x += dx/2; atom.y += dx*sqrt(3)/2
            elif abs(atom.x-(L-i*dl/2))<0.1 and abs(atom.y-i*dl*sqrt(3)/2)<tol: #and atom.y==dl*sqrt(3)/2:
                atom.symbol='F'
                if even:
                    atom.x += dx/2; atom.y -= dx*sqrt(3)/2
                else:
                    atom.x -= dx/2; atom.y += dx*sqrt(3)/2

def det(atom, pt1, pt2):
    return (pt1[0]-atom.x)*(pt2[1]-atom.y) - (pt1[1]-atom.y)*(pt2[0]-atom.x)


def arrange_conveniently(slab):
    slab.rotate('z',-pi)
    dx=slab[293].x
    dy=slab[293].y
    slab.translate((-dx,-dy,0))
    for atom in slab:
        if det(atom, (0,0), huc_line[0]) > tol :
            atom.x+=cell[0][0]
    slab.translate((0,0,-vac+2))

def lose_atoms(slab):
    half_layer = [atom for atom in slab if atom.tag==1 and det(atom, huc_line[0], huc_line[1]) > tol ]
    ctr = centro(huc_line[0],huc_line[1],huc_vert)
    del slab[[atom.index for atom in half_layer if atom.y < ctr[1] and abs(atom.x-huc_line[1][0]) < tol ]]
    del slab[[atom.index for atom in half_layer if not atom.x < ctr[0] and abs(det(atom,ctr,huc_vert)) < tol ]]
    del slab[[atom.index for atom in slab if atom.tag==1 and atom.x < ctr[0] and abs(det(atom,ctr,huc_line[0])) < tol ]]
    #print [atom.index for atom in slab if atom.x < ctr[0] and abs(det(atom,ctr,huc_line[0])) < tol ]

def reconstruct(slab):
    half_layer = [atom for atom in slab if atom.tag==1 and det(atom, huc_line[0], huc_line[1]) > tol ]
    
    pivot = slab[297]
    dx=slab[151].x-pivot.x
    dy=slab[151].y-pivot.y
    l1 = [atom for atom in half_layer if abs(atom.y-pivot.y)<tol or atom.index==298 or atom.index ==299]
    for atom in l1:
        atom.x+=dx
        atom.y+=dy
    
    pivot = slab[300]
    dx=slab[157].x-pivot.x
    dy=slab[157].y-pivot.y
    pt1=pivot.position[:2]
    pt2=slab[305].position[:2]
    l2 = [atom for atom in half_layer if abs(det(atom,pt1,pt2)) < tol or atom.index==304 or atom.index ==310]
    for atom in l2:
        atom.x+=dx
        atom.y+=dy
    
    pivot = slab[328]
    dx=slab[181].x-pivot.x
    dy=slab[181].y-pivot.y
    pt1=pivot.position[:2]
    pt2=slab[302].position[:2]
    l3 = [atom for atom in half_layer if abs(det(atom,pt1,pt2)) < tol or atom.index==303 or atom.index ==309]
    for atom in l3:
        atom.x+=dx
        atom.y+=dy

def add_adatoms(slab):
    #left
    adlayer = diamond111('O', size=(3,3,1), a=2*lc, vacuum=vac) 
    #view(adlayer)
    pt1=adlayer[6].position[:2]
    pt2=adlayer[2].position[:2]
    half_layer = [atom for atom in adlayer if det(atom, pt1, pt2) < tol ]
    print type(half_layer)
    #half_layer = [atom for atom in adlayer if det(atom, huc_line[0], huc_line[1]) < 0.001*tol ]
    add_adsorbate(slab,half_layer,-2.0,offset=1)
    #right
    adlayer = diamond111('O', size=(3,3,1), a=2*lc, vacuum=vac)
    pt1=adlayer[1].position[:2]
    pt2=adlayer[3].position[:2]
    half_layer = [atom for atom in adlayer if det(atom, pt1, pt2) < 0 ]
    add_adsorbate(slab,half_layer,-2.0,position=(slab[273].x,slab[273].y))#,offset=2)

lc = 5.46	# experimental value = 5.43
dl = lc/sqrt(2)
nx = 7#input('input # of atoms along x: ')
ny = 7#input('input # of atoms along y: ')
nz = 7#input('input # of atoms along z (odd!): ')

vac = 13
slab = diamond111('Si', size=(nx,ny,nz+1), a=lc, vacuum=vac) # NOTE +1 because 1 will be replaced for saturation
cell=slab.get_cell()
arrange_conveniently(slab)
print len(slab)
view(slab)
#print cell
#print dl
saturate(slab)
del_layer(slab,1)
print len(slab)
view(slab)
del_corner_atom(slab)
color_layers(slab)
print len(slab)
view(slab)
dimerize(slab)
slab.append(Atom('Xe', (huc_line[0][0],huc_line[0][1],15)))
slab.append(Atom('Xe', (huc_line[1][0],huc_line[1][1],15)))
slab.append(Atom('Xe', (huc_vert[0],huc_vert[1],15)))
#print huc_line[0]
#print huc_line[1]
#print huc_vert
#ctr = centro(huc_line[0],huc_line[1],huc_vert)
#slab.append(Atom('Xe', (ctr[0], ctr[1], 25)))
lose_atoms(slab)
reconstruct(slab)
add_adatoms(slab)
del slab[[atom.index for atom in slab if atom.symbol=='Xe']]
color_layers(slab,restore=True)
print len(slab)
view(slab)
write('Si111.traj', slab)
