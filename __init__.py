# thctk initialization
#
#
#   thctk - python package for Theoretical Chemistry
#   Copyright (C) 2004-2007 Christoph Scheurer
#
#   This file is part of thctk.
#
#   thctk is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   thctk is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the Free Software
#   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""
    thctk - Theoretical Chemistry Toolkit
    Copyright (C) 2004-2007 Christoph Scheurer
"""

#
# define all subpackages, this is also used by setup.py
#
#__all__ = [ 'spectroscopy', 'QC', 'MD', 'QD', 'numeric', 'statistics',
#            'visualization', 'parallel', 'extensionTypes' ]

__all__=['curvilinear','curvilinear.numeric']

__revision__ = """<<revision>>"""
__version__ = '0.1'

license = __doc__ + """
    thctk is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
 
    thctk is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
 
    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
"""
licence = license
copying = license
COPYING = license

