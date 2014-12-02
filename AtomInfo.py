# AtomInfo
#
#
#   This file was taken from Christoph Scheureres thctk package.
#
#   This file is also:
#   Copyright (C) 2006 Mehdi Bounouar

"""
Frequently used element information for Quantum Chemistry
"""

def sortkeys(d):
    """ Returns the keys of dictionary d sorted by their values.
        this function could be useful.
    """
    items=d.items()
    backitems=[ [v[1],v[0]] for v in items]
    backitems.sort()
    return [ backitems[i][1] for i in range(0,len(backitems))]

Names = ['', 'Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron',
         'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Neon', 'Sodium',
         'Magnesium', 'Aluminium', 'Silicon', 'Phosphorus', 'Sulfur',
         'Chlorine', 'Argon', 'Potassium', 'Calcium', 'Scandium',
         'Titanium', 'Vanadium', 'Chromium', 'Manganese', 'Iron',
         'Cobalt', 'Nickel', 'Copper', 'Zinc', 'Gallium', 'Germanium',
         'Arsenic', 'Selenium', 'Bromine', 'Krypton', 'Rubidium',
         'Strontium', 'Yttrium', 'Zirconium', 'Niobium', 'Molybdenum',
         'Technetium', 'Ruthenium', 'Rhodium', 'Palladium', 'Silver',
         'Cadmium', 'Indium', 'Tin', 'Antimony', 'Tellurium',
         'Iodine', 'Xenon', 'Caesium', 'Barium', 'Lanthanum',
         'Cerium', 'Praseodymium', 'Neodymium', 'Promethium',
         'Samarium', 'Europium', 'Gadolinium', 'Terbium',
         'Dysprosium', 'Holmium', 'Erbium', 'Thulium', 'Ytterbium',
         'Lutetium', 'Hafnium', 'Tantalum', 'Tungsten', 'Rhenium',
         'Osmium', 'Iridium', 'Platinum', 'Gold', 'Mercury',
         'Thallium', 'Lead', 'Bismuth', 'Polonium', 'Astatine',
         'Radon', 'Francium', 'Radium', 'Actinium', 'Thorium',
         'Protactinium', 'Uranium', 'Neptunium', 'Plutonium',
         'Americium', 'Curium', 'Berkelium', 'Californium',
         'Einsteinium', 'Fermium', 'Mendelevium', 'Nobelium',
         'Lawrencium', 'Unnilquadium', 'Unnilpentium', 'Unnilhexium']


Symbols = ['X',  'H',  'He', 'Li', 'Be',
           'B',  'C',  'N',  'O',  'F',
           'Ne', 'Na', 'Mg', 'Al', 'Si',
           'P',  'S',  'Cl', 'Ar', 'K',
           'Ca', 'Sc', 'Ti', 'V',  'Cr',
           'Mn', 'Fe', 'Co', 'Ni', 'Cu',
           'Zn', 'Ga', 'Ge', 'As', 'Se',
           'Br', 'Kr', 'Rb', 'Sr', 'Y',
           'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
           'Rh', 'Pd', 'Ag', 'Cd', 'In',
           'Sn', 'Sb', 'Te', 'I',  'Xe',
           'Cs', 'Ba', 'La', 'Ce', 'Pr',
           'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
           'Tb', 'Dy', 'Ho', 'Er', 'Tm',
           'Yb', 'Lu', 'Hf', 'Ta', 'W',
           'Re', 'Os', 'Ir', 'Pt', 'Au',
           'Hg', 'Tl', 'Pb', 'Bi', 'Po',
           'At', 'Rn', 'Fr', 'Ra', 'Ac',
           'Th', 'Pa', 'U',  'Np', 'Pu',
           'Am', 'Cm', 'Bk', 'Cf', 'Es',
           'Fm', 'Md', 'No', 'Lw']

Symbol = Symbols

SymbolToNumber = {}
for i in range(len(Symbols)):
    SymbolToNumber[Symbols[i]] = i
    SymbolToNumber[Symbols[i].lower()] = i

def sym2no(a):
    k = a.strip().lower()
    if len(k) == 1:
        n = SymbolToNumber[k]
    else:
        try:
            n = SymbolToNumber[k[:2]]
        except KeyError:
            n = SymbolToNumber[k[0]]
    return n

Mass = [
   0.000000000, # X
   1.007825037, # H
   4.002603250, # He
   7.016004500, # Li
   9.012182500, # Be
  11.009305300, # B
  12.000000000, # C
  14.003074008, # N
  15.994914640, # O
  18.998403250, # F
  19.992439100, # Ne
  22.989769700, # Na
  23.985045000, # Mg
  26.981541300, # Al
  27.976928400, # Si
  30.973763400, # P
  31.972071800, # S
  34.968852729, # Cl
  39.962383100, # Ar
  38.963707900, # K
  39.962590700, # Ca
  44.955913600, # Sc
  47.947946700, # Ti
  50.943962500, # V
  51.940509700, # Cr
  54.938046300, # Mn
  55.934939300, # Fe
  58.933197800, # Co
  57.935347100, # Ni
  62.929599200, # Cu
  63.929145400, # Zn
  68.925580900, # Ga
  73.921178800, # Ge
  74.921595500, # As
  79.916520500, # Se
  78.918336100, # Br
  83.911506400, # Kr
  85.4678,      # Rb
  87.6200,      # Sr
  88.9059,      # Y
  91.2200,      # Zr
  92.9064,      # Nb
  95.9400,      # Mo
  98.0000,      # Tc
 101.0700,      # Ru
 102.9055,      # Rh
 106.4000,      # Pd
 107.8680,      # Ag
 112.4100,      # Cd
 114.8200,      # In
 118.6900,      # Sn
 121.7500,      # Sb
 127.6000,      # Te
 126.9045,      # I
 131.3000,      # Xe
 132.9054,      # Cs
 137.3300,      # Ba
 138.9055,      # La
 140.1200,      # Ce
 140.9077,      # Pr
 144.2400,      # Nd
 145.0000,      # Pm
 150.4000,      # Sm
 151.9600,      # Eu
 157.2500,      # Gd
 158.9254,      # Tb
 162.5000,      # Dy
 164.9304,      # Ho
 167.2600,      # Er
 168.9342,      # Tm
 173.0400,      # Yb
 174.9670,      # Lu
 178.4900,      # Hf
 180.9479,      # Ta
 183.8500,      # W
 186.2070,      # Re
 190.2000,      # Os
 192.2200,      # Ir
 195.0900,      # Pt
 196.9665,      # Au
 200.5900,      # Hg
 204.3700,      # Tl
 207.2000,      # Pb
 208.9804,      # Bi
 209.0000,      # Po
 210.0000,      # At
 222.0000,      # Rn
 223.0000,      # Fr
 226.0254,      # Ra
 227.0278,      # Ac
 232.0381,      # Th
 231.0359,      # Pa
 238.0290,      # U
 237.0482,      # Np
 244.0000,      # Pu
 243.0000,      # Am
 247.0000,      # Cm
 247.0000,      # Bk
 251.0000,      # Cf
 254.0000,      # Es
 257.0000,      # Fm
 258.0000,      # Md
 259.0000,      # No
 260.0000       # Lw
  ]

IUPMass=[
0.000000000, # X
1.00794   , # H    
4.002602  , # He
6.941     , # Li
9.012182  , # Be
10.811    , # B
12.0110   , # C
14.0067   , # N
15.9994   , # O
18.9984032, # F
20.1797   , # Ne
22.989770 , # Na
24.3050   , # Mg
26.981538 , # Al
28.0855   , # Si
30.973761 , # P
32.065    , # S
35.453    , # Cl
39.948    , # Ar
39.0983   , # K
40.078    , # Ca
44.955910 , # Sc
47.867    , # Ti
50.9415   , # V
51.9961   , # Cr
54.938049 , # Mn
55.845    , # Fe
58.933200 , # Co
58.6934   , # Ni
63.546    , # Cu
65.409    , # Zn
69.723    , # Ga
72.64     , # Ge
74.92160  , # As
78.96     , # Se
79.904    , # Br
83.798    , # Kr
85.4678   , # Rb
87.62     , # Sr
88.90585  , # Y
91.224    , # Zr
92.90638  , # Nb
95.94       # Mo
]

Covalent_Radii = [
 0.20, # X
 0.32, # H
 0.93, # He
 1.23, # Li
 0.90, # Be
 0.82, # B
 0.77, # C
 0.75, # N
 0.73, # O
 0.72, # F
 0.71, # Ne
 1.54, # Na
 1.36, # Mg
 1.18, # Al
 1.11, # Si
 1.06, # P
 1.02, # S
 0.99, # Cl
 0.98, # Ar
 2.03, # K
 1.74, # Ca
 1.44, # Sc
 1.32, # Ti
 1.22, # V
 1.18, # Cr
 1.17, # Mn
 1.17, # Fe
 1.16, # Co
 1.15, # Ni
 1.17, # Cu
 1.25, # Zn
 1.26, # Ga
 1.22, # Ge
 1.20, # As
 1.16, # Se
 1.14, # Br
 1.89, # Kr
 2.16, # Rb
 1.91, # Sr
 1.62, # Y
 1.45, # Zr
 1.34, # Nb
 1.30, # Mo
 1.27, # Tc
 1.25, # Ru
 1.25, # Rh
 1.28, # Pd
 1.34, # Ag
 1.41, # Cd
 1.44, # In
 1.41, # Sn
 1.40, # Sb
 1.36, # Te
 1.33, # I
 1.31, # Xe
 2.35, # Cs
 1.98, # Ba
 1.25, # La
 1.65, # Ce
 1.65, # Pr
 1.64, # Nd
 1.63, # Pm
 1.62, # Sm
 1.85, # Eu
 1.61, # Gd
 1.59, # Tb
 1.59, # Dy
 1.58, # Ho
 1.57, # Er
 1.56, # Tm
 1.70, # Yb
 1.56, # Lu
 1.44, # Hf
 1.34, # Ta
 1.30, # W
 1.28, # Re
 1.26, # Os
 1.27, # Ir
 1.30, # Pt
 1.34, # Au
 1.49, # Hg
 1.48, # Tl
 1.47, # Pb
 1.46, # Bi
 1.53, # Po
 1.47, # At
 None, # Rn
 None, # Fr
 None, # Ra
 None, # Ac
 1.65, # Th
 None, # Pa
 1.42, # U
 None, # Np
 None, # Pu
 None, # Am
 None, # Cm
 None, # Bk
 None, # Cf
 None, # Es
 None, # Fm
 None, # Md
 None, # No
 None] # Lw

 
