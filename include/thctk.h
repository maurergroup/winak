/*
 * thctk.h
 *
 *   thctk - python package for Theoretical Chemistry
 *   Copyright (C) 2002-2006 Christoph Scheurer
 *
 *   This file is part of thctk.
 *
 *   thctk is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   thctk is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */
#ifndef THCTK_H_SEEN
#define THCTK_H_SEEN
#ifdef __cplusplus
extern "C" {
#endif

#ifdef THCTK_NUMBACKEND
#if THCTK_NUMBACKEND == 0
#include "Numeric/arrayobject.h"
#endif
#if THCTK_NUMBACKEND == 1
#include "numpy/arrayobject.h"
#include "numpy/noprefix.h"
#endif
#else
#include "Numeric/arrayobject.h"
#endif

#ifndef max
#define max(a,b) ((a) >= (b) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) ((a) <= (b) ? (a) : (b))
#endif
#ifndef abs
#define abs(x) ((x) >= 0 ? (x) : -(x))
#endif

#ifndef UINT8
#define UINT8 unsigned char
#endif

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

/* the following are macros to put values of preprocessor macros in
 * doc-strings
 */
#define STRINGIFY(x)    #x
#define TOSTRING(x)     STRINGIFY(x)

#if defined(PY_VERSION_HEX) && PY_VERSION_HEX < 0x02050000
#ifndef PY_SSIZE_T_MIN
    typedef int Py_ssize_t;
#endif
#endif

#ifndef THCTK_PRIVATE
#define THCTK_PRIVATE   static
#define THCTKDOC(M,F)   static char M ## _ ## F ## __doc__[] 
#define THCTKDEF(M,F)   { #F , (PyCFunction) M ## _ ## F, METH_VARARGS, M ## _ ## F ## __doc__},
#define THCTKFUN(M,F)   static PyObject * M ## _ ## F (PyObject * unused, PyObject * args)
#define THCTKMOD(M)     PyMODINIT_FUNC init ## M (void) { PyObject *m; \
        m = Py_InitModule3(#M , M ## _methods, M ## _module_documentation); \
        if (PyModule_AddIntConstant(m, "numericBackend", THCTK_NUMBACKEND)) \
            Py_FatalError("can't set variable numericBackend"); \
        import_array(); \
        if (PyErr_Occurred()) { Py_FatalError("can't initialize module M "); }}
#endif

#ifdef __cplusplus
}
#endif
#endif /* !THCTK_H_SEEN */
