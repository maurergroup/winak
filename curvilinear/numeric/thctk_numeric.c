/*
 * thctk.numeric: thctk_numeric.c
 *
 *   thctk - python package for Theoretical Chemistry
 *   Copyright (C) 2002-2007 Christoph Scheurer
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
#include <omp.h>
#include <math.h>
#include "colamd.h"

#ifdef THCTK_INTERFACE
#include <Python.h>
#include "thctk.h"
/*  Every function in this module needs a doc string
 *
 *      THCTKDOC(module,function) = "";
 *
 *  and a function body that has to start like
 *
 *      THCTKFUN(module,function)
 *
 *  The calling arguments are passed in the PyObject *args
 *
 *  The module name 'module' has to match the name used in the
 *  setup.py list ext_modules.
 *
 *  The following is sample code for the usage of the macros in
 *  "thctk.h"

THCTKDOC(module,template) = "";

THCTKFUN(module,template)
{
    if (!PyArg_ParseTuple(args, "", )) return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}

 */


THCTKDOC(_numeric, daxpy_p) = "x = daxpy_p(a, x, y, p, [inverse])\n"
"Compute a*x+p(y) or a*x+p^-1(y).\n";

THCTKFUN(_numeric, daxpy_p)
{

    PyArrayObject *xa, *ya, *pa;
    double a, *x, *y;
    int i, n, *p, inverse=0;

    if (!PyArg_ParseTuple(args, "dO!O!O!|i", &a, &PyArray_Type, &xa,
        &PyArray_Type, &ya, &PyArray_Type, &pa, &inverse)) return NULL;

    n = xa->dimensions[0];

    x = (double *) xa->data;
    y = (double *) ya->data;
    p = (int *) pa->data;

    if (a != 1)  for (i=0; i<n; i++) x[i] *= a;

    if (inverse) for (i=0; i<n; i++) x[p[i]] += y[i];
    else         for (i=0; i<n; i++) x[i] += y[p[i]];

    Py_INCREF(xa);
    return (PyObject *) xa;

}


THCTKDOC(_numeric, copyArray) = "b = copyArray(a, b)\n"
"copy min(len(a), len(b)) elements from array a into array b.\n"
"The arrays have to be contiguous and of the same typecode.\n";

THCTKFUN(_numeric, copyArray)
{

    PyArrayObject *a, *b;
    int i, na=1, nb=1, n;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a, &PyArray_Type, &b))
        return NULL;

    if (a->descr->type_num != b->descr->type_num) {
        PyErr_SetString(PyExc_MemoryError, "typecode mismatch");
        return NULL;
    }
    if (!PyArray_ISCONTIGUOUS(b)) {
        PyErr_SetString(PyExc_MemoryError, "destination array is not contiguous");
        return NULL;
    }
    if (!PyArray_ISCONTIGUOUS(a)) {
        PyErr_SetString(PyExc_MemoryError, "source array is not contiguous");
        return NULL;
    }

    for (i=0; i<(a->nd); i++) na *= a->dimensions[i];
    for (i=0; i<(b->nd); i++) nb *= b->dimensions[i];
    n = min(na, nb) * (a->descr->elsize);

    memcpy((void *) b->data, (const void *) a->data, n);

    Py_INCREF(b);
    return (PyObject *) b;

}


THCTKDOC(_numeric, colamd) = "p = colamd(rows, jb, ib)\n"
"jb and ib describe the nonzero structure of matrix B in sparse column format\n"
"p is the permutation produced by the colamd ordering scheme\n";

THCTKFUN(_numeric, colamd)
{

    PyArrayObject *p=NULL, *jb, *ib;
    double *knobs=NULL;
    int columns, rows, nnz, *a=NULL, n, stats[COLAMD_STATS], ok, i;

    if (!PyArg_ParseTuple(args, "iO!O!", &rows, &PyArray_Type, &jb,
        &PyArray_Type, &ib)) return NULL;

    columns = ib->dimensions[0] - 1;
#if THCTK_NUMBACKEND == 1   // NumPy
    nnz = *((npy_int32 *) PyArray_GETPTR1(ib, columns));
#else
    nnz = ((int *) ib->data)[columns];
#endif

    if (! (p = (PyArrayObject *) PyArray_CopyFromObject((PyObject *) ib,
        PyArray_INT, 0, 1)) ) return NULL;

    n = colamd_recommended(nnz, rows, columns);
    if (! (a = (int *) malloc(n*sizeof(int))) ) return NULL;

#if THCTK_NUMBACKEND == 1   // NumPy
    for (i=0; i<nnz; i++) {
        a[i] = ((npy_int32 *) jb->data)[i];
    }
#else
    a = (int *) memcpy((void *) a, (const void *) jb->data, nnz*sizeof(int));
#endif

    ok = colamd(rows, columns, n, a, (int *) p->data, knobs, stats);

    if (a) free(a);

    if (ok) {
        return PyArray_Return(p);
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }

}

#endif

THCTK_PRIVATE
int amux_CSR(const double *m, const int *mj, const int *mi, 
    const double *xx, double *yy, int n, int offset) {

    int i, k;
    #pragma omp parallel for private(i,k)\
        shared(m, mj, mi, xx, yy, n, offset)
    for (i=offset; i<n; i++) {
        register double t=0;
        for (k=mi[i]; k<mi[i+1]; k++) t += m[k] * xx[mj[k]];
        yy[i] = t;
    }
    return 1;
}

THCTK_PRIVATE
int amux_CSR_perm(const double *m, const int *mj, const int *mi, 
    const double *xx, double *yy, const int *p, int n, int offset) {

    int i, k;
    #pragma omp parallel for private(i,k)\
        shared(m, mj, mi, xx, yy, p, n, offset)
    for (i=offset; i<n; i++) {
        register double t=0;
        for (k=mi[i]; k<mi[i+1]; k++) t += m[k] * xx[p[mj[k]]];
        yy[i] = t;
    }
    return 1;
}

#ifdef THCTK_INTERFACE
THCTKDOC(_numeric, amux_CSR) =
"y = amux_CSR(a, aj, ai, x, y, [offset, p])\n"
"Compute y = a x  or y = a p(x), where p is an optional permutation and\n"
"the matrix a is given in CSR format. If offset=0 (default) the indices in\n"
"aj, ai, and p are in C convention (starting at 0) otherwise Fortran convention\n"
"(starting at 1) is assumed for aj and ai. Fortran convention is also assumed\n"
"for the values in p if offset>0.\n"
"numpy.int32 data type must be used for p\n";

THCTKFUN(_numeric, amux_CSR)
{

    PyArrayObject *a, *aj, *ai, *x, *y, *p=NULL;
    int n, offset=0;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!|iO!", &PyArray_Type, &a,
        &PyArray_Type, &aj, &PyArray_Type, &ai, &PyArray_Type, &x,
        &PyArray_Type, &y, &offset, &PyArray_Type, &p)) return NULL;

    if (offset != 0) offset = 1;
    n = ai->dimensions[0] - 1 + offset;

    if (p) { 
         if (!amux_CSR_perm(((double *) a->data) - offset, 
                        ((int *) aj->data) - offset, 
                        ((int *) ai->data) - offset, 
                        ((double *) x->data) - offset,
                        ((double *) y->data) - offset, 
                        ((int *) p->data) - offset,
                        n, offset )) return NULL;
        }
    else {
         if (! amux_CSR(((double *) a->data) - offset, 
                        ((int *) aj->data) - offset, 
                        ((int *) ai->data) - offset, 
                        ((double *) x->data) - offset,
                        ((double *) y->data) - offset, 
                        n, offset)) return NULL;
        }

    Py_INCREF(y);
    return (PyObject *) y;

}

#endif

THCTK_PRIVATE
int amux_CSR_re_complex(const double *m, const int *mj, const int *mi, 
    const double *xx, double *yy, int n, int offset) {

    int i, k;
    #pragma omp parallel for private(i,k)\
        shared(m, mj, mi, xx, yy, n, offset)
    for (i=offset; i<n; i++) {
        register double t=0, u=0;
        for (k=mi[i]; k<mi[i+1]; k++){
            t += m[k] * xx[2*mj[k]];
            u += m[k] * xx[2*mj[k]+1];
            }
        yy[2*i]   = t;
        yy[2*i+1] = u;
    }
    return 1;
}

THCTK_PRIVATE
int amux_CSR_re_complex_perm(const double *m, const int *mj, const int *mi, 
    const double *xx, double *yy, const int *p, int n, int offset) {

    int i, k;
    #pragma omp parallel for private(i,k)\
        shared(m, mj, mi, xx, yy, n, offset)
    for (i=offset; i<n; i++) {
        register double t=0, u=0;
        for (k=mi[i]; k<mi[i+1]; k++){
             t += m[k] * xx[2*p[mj[k]]];
             u += m[k] * xx[2*p[mj[k]]+1];
             }
        yy[2*i]   = t;
        yy[2*i+1] = u;
    }
    return 1;
}

#ifdef THCTK_INTERFACE
THCTKDOC(_numeric, amux_CSR_re_complex) =
"y = amux_CSR(a, aj, ai, x, y, [offset, p])\n"
"same as amux_CSR but for complex valued matrix a AND vectors x and y.\n";

THCTKFUN(_numeric, amux_CSR_re_complex)
{

    PyArrayObject *a, *aj, *ai, *x, *y, *p=NULL;
    int n, offset=0;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!|iO!", &PyArray_Type, &a,
        &PyArray_Type, &aj, &PyArray_Type, &ai, &PyArray_Type, &x,
        &PyArray_Type, &y, &offset, &PyArray_Type, &p)) return NULL;

    if (offset != 0) offset = 1;
    n = ai->dimensions[0] - 1 + offset;

    if (p) {
         if (! amux_CSR_re_complex_perm(((double *) a->data) - offset, 
                        ((int *) aj->data) - offset, 
                        ((int *) ai->data) - offset, 
                        ((double *) x->data) - 2*offset,
                        ((double *) y->data) - 2*offset, 
                        ((int *) p->data) - offset,
                        n, offset)) return NULL;
    } else {
         if (! amux_CSR_re_complex(((double *) a->data) - offset, 
                        ((int *) aj->data) - offset, 
                        ((int *) ai->data) - offset, 
                        ((double *) x->data) - 2*offset,
                        ((double *) y->data) - 2*offset, 
                        n, offset)) return NULL;
    }

    Py_INCREF(y);
    return (PyObject *) y;

}

#endif

THCTK_PRIVATE
int amux_CSR_complex(const double *m, const int *mj, const int *mi, 
    const double *xx, double *yy, int n, int offset) {

    int i, k;
    #pragma omp parallel for private(i,k)\
        shared(m, mj, mi, xx, yy, n, offset)
    for (i=offset; i<n; i++) {
        register double t=0, u=0;
        for (k=mi[i]; k<mi[i+1]; k++){
             t += m[2*k] *   xx[2*mj[k]];
             t -= m[2*k+1] * xx[2*mj[k]+1];
             u += m[2*k] *   xx[2*mj[k]+1];
             u += m[2*k+1] * xx[2*mj[k]];
             }
        yy[2*i]   = t;
        yy[2*i+1] = u;
    }
    return 1;
}

THCTK_PRIVATE
int amux_CSR_complex_perm(const double *m, const int *mj, const int *mi, 
    const double *xx, double *yy, const int *p, int n, int offset) {

    int i, k;
    #pragma omp parallel for private(i,k)\
        shared(m, mj, mi, xx, yy, n, offset)
    for (i=offset; i<n; i++) {
        register double t=0, u=0;
        for (k=mi[i]; k<mi[i+1]; k++){
             t += m[2*k] *   xx[2*p[mj[k]]];
             t -= m[2*k+1] * xx[2*p[mj[k]]+1];
             u += m[2*k] *   xx[2*p[mj[k]]+1];
             u += m[2*k+1] * xx[2*p[mj[k]]];
             }
        yy[2*i]   = t;
        yy[2*i+1] = u;
    }
    return 1;
}

#ifdef THCTK_INTERFACE
THCTKDOC(_numeric, amux_CSR_complex) =
"y = amux_CSR(a, aj, ai, x, y, [offset, p])\n"
"same as amux_CSR but for complex valued matrix a AND vectors x and y.\n";

THCTKFUN(_numeric, amux_CSR_complex)
{

    PyArrayObject *a, *aj, *ai, *x, *y, *p=NULL;
    int n, offset=0;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!|iO!", &PyArray_Type, &a,
        &PyArray_Type, &aj, &PyArray_Type, &ai, &PyArray_Type, &x,
        &PyArray_Type, &y, &offset, &PyArray_Type, &p)) return NULL;

    if (offset != 0) offset = 1;
    n = ai->dimensions[0] - 1 + offset;

    if (p) {
         if (! amux_CSR_complex_perm(((double *) a->data) - 2*offset, 
                        ((int *) aj->data) - offset, 
                        ((int *) ai->data) - offset, 
                        ((double *) x->data) - 2*offset,
                        ((double *) y->data) - 2*offset, 
                        ((int *) p->data) - offset,
                        n, offset)) return NULL;
    } else {
         if (! amux_CSR_complex(((double *) a->data) - 2*offset, 
                        ((int *) aj->data) - offset, 
                        ((int *) ai->data) - offset, 
                        ((double *) x->data) - 2*offset,
                        ((double *) y->data) - 2*offset, 
                        n, offset)) return NULL;
    }

    Py_INCREF(y);
    return (PyObject *) y;

}

#endif

THCTK_PRIVATE
int amux_CSRd(const double *m, const double *d, const int *mj, 
    const int *mi, const double *xx, double *yy, int n, int offset) {

    int i, k;
    #pragma omp parallel private(i,k)\
        shared(m, d, mj, mi, xx, yy, n, offset)
        {
    #pragma omp for
    for (i=offset; i<n; i++) yy[i] = d[i] * xx[i];
    #pragma omp for
    for (i=offset; i<n; i++) {
        register double t=0;
        for (k=mi[i]; k<mi[i+1]; k++) t += m[k] * xx[mj[k]];
        yy[i] += t;
    }
    }
    return 1;
}

THCTK_PRIVATE
int amux_CSRd_perm(const double *m, const double *d, const int *mj, 
    const int *mi, const double *xx, double *yy, const int *p, int n, 
    int offset) {

    int i, k;
    #pragma omp parallel private(i,k)\
        shared(m, d, mj, mi, xx, yy, p, n, offset)
        {
    #pragma omp for
    for (i=offset; i<n; i++) yy[i] = d[i] * xx[p[i]];
    #pragma omp for
    for (i=offset; i<n; i++) {
        register double t=0;
        for (k=mi[i]; k<mi[i+1]; k++) t += m[k] * xx[p[mj[k]]];
        yy[i] += t;
    }
    }
    return 1;
}


#ifdef THCTK_INTERFACE
THCTKDOC(_numeric, amux_CSRd) =
"y = amux_CSRd(a, ad, aj, ai, x, y, [offset, p])\n"
"Compute y = a x  or y = a p(x), where p is an optional permutation and the\n"
"symmetric matrix a is given in CSRd format. If offset=0 (default) the indices\n"
"in aj, ai, and p are in C convention (starting at 0) otherwise Fortran\n"
"convention (starting at 1) is assumed for aj and ai. Fortran convention is\n"
"also assumed for the values in p if offset>0.\n";

THCTKFUN(_numeric, amux_CSRd)
{

    PyArrayObject *a, *ad, *aj, *ai, *x, *y, *p=NULL;
    int n, offset=0;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!|iO!", &PyArray_Type, &a,
        &PyArray_Type, &ad, &PyArray_Type, &aj, &PyArray_Type, &ai,
        &PyArray_Type, &x, &PyArray_Type, &y, &offset, &PyArray_Type, &p))
        return NULL;

    if (offset != 0) offset = 1;
    n = ai->dimensions[0] - 1 + offset;

    if (p) {
        if (! amux_CSRd_perm(((double *) a->data) - offset,
                        ((double *) ad->data) - offset,
                        ((int *) aj->data) - offset,
                        ((int *) ai->data) - offset,
                        ((double *) x->data) - offset,
                        ((double *) y->data) - offset,
                        ((int *) p->data) - offset,
                        n, offset)) return NULL;
    } else {
        if (! amux_CSRd(((double *) a->data) - offset,
                        ((double *) ad->data) - offset,
                        ((int *) aj->data) - offset,
                        ((int *) ai->data) - offset,
                        ((double *) x->data) - offset,
                        ((double *) y->data) - offset,
                        n, offset)) return NULL;
    }

    Py_INCREF(y);
    return (PyObject *) y;

}

#endif

THCTK_PRIVATE
int amux_CSRd_complex(const double *m, const double *d, const int *mj, 
    const int *mi, const double *xx, double *yy, int n, int offset) {

    int i, k;
    #pragma omp parallel private(i,k)\
        shared(m, d, mj, mi, xx, yy, n, offset)
        {
    #pragma omp for
    for (i=offset; i<n; i++) {
        yy[2*i]   = d[2*i] *   xx[2*i] - d[2*i+1] * xx[2*i+1];
        yy[2*i+1] = d[2*i+1] * xx[2*i] + d[2*i] *   xx[2*i+1];
    }
    #pragma omp for
    for (i=offset; i<n; i++) {
        register double t=0, u=0;
        for (k=mi[i]; k<mi[i+1]; k++){
             t += m[2*k] *   xx[2*mj[k]];
             t -= m[2*k+1] * xx[2*mj[k]+1];
             u += m[2*k] *   xx[2*mj[k]+1];
             u += m[2*k+1] * xx[2*mj[k]];
             }
        yy[2*i]   += t;
        yy[2*i+1] += u;
    }
    }
    return 1;
}

THCTK_PRIVATE
int amux_CSRd_complex_perm(const double *m, const double *d, 
    const int *mj, const int *mi, const double *xx, double *yy, 
    const int *p, int n, int offset) {

    int i, k;
    #pragma omp parallel private(i,k)\
        shared(m, d, mj, mi, xx, yy, n, offset)
        {
    #pragma omp for
    for (i=offset; i<n; i++) {
        yy[2*i]   = d[2*i] *   xx[2*p[i]] - d[2*i+1] * xx[2*p[i]+1];
        yy[2*i+1] = d[2*i+1] * xx[2*p[i]] + d[2*i] *   xx[2*p[i]+1];
    }
    #pragma omp for
    for (i=offset; i<n; i++) {
        register double t=0, u=0;
        for (k=mi[i]; k<mi[i+1]; k++){
             t += m[2*k] *   xx[2*p[mj[k]]];
             t -= m[2*k+1] * xx[2*p[mj[k]]+1];
             u += m[2*k] *   xx[2*p[mj[k]]+1];
             u += m[2*k+1] * xx[2*p[mj[k]]];
             }
        yy[2*i]   += t;
        yy[2*i+1] += u;
    }
    }
    return 1;
}

#ifdef THCTK_INTERFACE
THCTKDOC(_numeric, amux_CSRd_complex) =
"y = amux_CSRd(a, ad, aj, ai, x, y, [offset, p])\n"
"same as amux_CSRd but for complex valued matrix a, ad\n"
"AND vectors x and y.\n";

THCTKFUN(_numeric, amux_CSRd_complex)
{

    PyArrayObject *a, *ad, *aj, *ai, *x, *y, *p=NULL;
    int n, offset=0;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!|iO!", &PyArray_Type, &a,
        &PyArray_Type, &ad, &PyArray_Type, &aj, &PyArray_Type, &ai,
        &PyArray_Type, &x, &PyArray_Type, &y, &offset, &PyArray_Type, &p))
        return NULL;

    if (offset != 0) offset = 1;
    n = ai->dimensions[0] - 1 + offset;

    if (p) {
        if (! amux_CSRd_complex_perm(((double *) a->data) - 2*offset,
                        ((double *) ad->data) - 2*offset,
                        ((int *) aj->data) - offset,
                        ((int *) ai->data) - offset,
                        ((double *) x->data) - 2*offset,
                        ((double *) y->data) - 2*offset,
                        ((int *) p->data) - offset,
                        n, offset)) return NULL;
    } else {
        if (! amux_CSRd_complex(((double *) a->data) - 2*offset,
                        ((double *) ad->data) - 2*offset,
                        ((int *) aj->data) - offset,
                        ((int *) ai->data) - offset,
                        ((double *) x->data) - 2*offset,
                        ((double *) y->data) - 2*offset,
                        n, offset)) return NULL;
    }

    Py_INCREF(y);
    return (PyObject *) y;

}

#endif

THCTK_PRIVATE
int amux_CSRd_re_complex(const double *m, const double *d, const int *mj, 
    const int *mi, const double *xx, double *yy, int n, int offset) {

    int i, k;
    #pragma omp parallel private(i,k)\
        shared(m, d, mj, mi, xx, yy, n, offset)
        {
    #pragma omp for
    for (i=offset; i<n; i++) {
        yy[2*i]   = d[i] * xx[2*i];
        yy[2*i+1] = d[i] * xx[2*i+1];
    }
    #pragma omp for
    for (i=offset; i<n; i++) {
        register double t=0, u=0;
        for (k=mi[i]; k<mi[i+1]; k++){
            t += m[k] * xx[2*mj[k]];
            u += m[k] * xx[2*mj[k]+1];
            }
        yy[2*i]   += t;
        yy[2*i+1] += u;
    }
    }
    return 1;
}

THCTK_PRIVATE
int amux_CSRd_re_complex_perm(const double *m, const double *d, 
    const int *mj, const int *mi, const double *xx, double *yy, 
    const int *p, int n, int offset) {

    int i, k;
    #pragma omp parallel private(i,k)\
        shared(m, d, mj, mi, xx, yy, n, offset)
        {
    #pragma omp for
    for (i=offset; i<n; i++) {
        yy[2*i]   = d[i] * xx[2*p[i]];
        yy[2*i+1] = d[i] * xx[2*p[i]+1];
    }
    #pragma omp for
    for (i=offset; i<n; i++) {
        register double t=0, u=0;
        for (k=mi[i]; k<mi[i+1]; k++){
            t += m[k] * xx[2*p[mj[k]]];
            u += m[k] * xx[2*p[mj[k]]+1];
            }
        yy[2*i]   += t;
        yy[2*i+1] += u;
    }
    }
    return 1;
}

#ifdef THCTK_INTERFACE
THCTKDOC(_numeric, amux_CSRd_re_complex) =
"y = amux_CSRd(a, ad, aj, ai, x, y, [offset, p])\n"
"same as amux_CSRd but for complex valued matrix a, ad\n"
"AND vectors x and y.\n";

THCTKFUN(_numeric, amux_CSRd_re_complex)
{

    PyArrayObject *a, *ad, *aj, *ai, *x, *y, *p=NULL;
    int n, offset=0;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!|iO!", &PyArray_Type, &a,
        &PyArray_Type, &ad, &PyArray_Type, &aj, &PyArray_Type, &ai,
        &PyArray_Type, &x, &PyArray_Type, &y, &offset, &PyArray_Type, &p))
        return NULL;

    if (offset != 0) offset = 1;
    n = ai->dimensions[0] - 1 + offset;

    if (p) {
        if (! amux_CSRd_re_complex_perm(((double *) a->data) - offset,
                        ((double *) ad->data) - offset,
                        ((int *) aj->data) - offset,
                        ((int *) ai->data) - offset,
                        ((double *) x->data) - 2*offset,
                        ((double *) y->data) - 2*offset,
                        ((int *) p->data) - offset,
                        n, offset)) return NULL;
    } else {
        if (! amux_CSRd_re_complex(((double *) a->data) - offset,
                        ((double *) ad->data) - offset,
                        ((int *) aj->data) - offset,
                        ((int *) ai->data) - offset,
                        ((double *) x->data) - 2*offset,
                        ((double *) y->data) - 2*offset,
                        n, offset)) return NULL;
    }

    Py_INCREF(y);
    return (PyObject *) y;

}


THCTK_PRIVATE
void inv_L_x_cd(int n, const double *l, const double *ldiag, const int *jl,
    const int *il, double *r, int offset) {

    /* suffix _cd stands for column storage with separate diagonal */

    int j, m;

    if (offset != 0) {  /* Fortran convention, indices start at 1 */
        offset = 1;
        l--;
        ldiag--;
        il--;
        jl--;
        r--;
        m = n;
    } else {
        m = n - 1;
    }

    /* Solve L x = r and store the result in r. */

    for (j=offset; j<=m; j++) {
        register int k;
        register double temp = r[j]/ldiag[j];
        for (k=jl[j]; k<jl[j+1]; k++) r[il[k]] -= l[k]*temp;
        r[j] = temp;
    }
}


THCTKDOC(_numeric, inv_L_x_cd) =
"x = inv_L_x_cd(l, ldiag, jl, il, r, offset)\n\n"
"Compute x = L^-1 r where the strict lower triangle of L is stored in l in\n"
"compressed column storage with row indices in il and column pointers in jl;\n"
"the diagonal of L is stored separately in ldiag. The vector r is overwritten\n"
"by the solution x on return.\n"
"If offset=0 the indices are in C convention (starting at 0) otherwise Fortran\n"
"convention (starting at 1) is assumed\n";

THCTKFUN(_numeric, inv_L_x_cd)
{

    PyArrayObject *l, *ldiag, *jl, *il, *r;
    int n, offset;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!i", &PyArray_Type, &l,
        &PyArray_Type, &ldiag, &PyArray_Type, &jl, &PyArray_Type, &il,
        &PyArray_Type, &r, &offset)) return NULL;

    n = ldiag->dimensions[0];

    inv_L_x_cd(n, (double *) l->data, (double *) ldiag->data,
        (int *) jl->data, (int *) il->data, (double *) r->data, offset);

    Py_INCREF(r);
    return (PyObject *) r;

}


THCTK_PRIVATE
void inv_Lt_x_cd(int n, const double *l, const double *ldiag, const int *jl,
    const int *il, double *r, int offset) {

    /* suffix _cd stands for column storage with separate diagonal */

    int j, m;

    if (offset != 0) {  /* Fortran convention, indices start at 1 */
        offset = 1;
        l--;
        ldiag--;
        il--;
        jl--;
        r--;
        m = n;
    } else {
        m = n - 1;
    }

    /* Solve L' x = r and store the result in r. */

    r[m] /= ldiag[m];
    for (j=m-1; j>=offset; j--) {
        register int k;
        register double temp = 0;
        for (k=jl[j]; k<jl[j+1]; k++) temp += l[k]*r[il[k]];
        r[j] -= temp;
        r[j] /= ldiag[j];
    }
}


THCTKDOC(_numeric, inv_Lt_x_cd) =
"x = inv_L_x_cd(l, ldiag, jl, il, r, offset)\n\n"
"Compute x = L^-1 r where the strict lower triangle of L is stored in l in\n"
"compressed column storage with row indices in il and column pointers in jl;\n"
"the diagonal of L is stored separately in ldiag. The vector r is overwritten\n"
"by the solution x on return.\n"
"If offset=0 the indices are in C convention (starting at 0) otherwise Fortran\n"
"convention (starting at 1) is assumed\n";

THCTKFUN(_numeric, inv_Lt_x_cd)
{

    PyArrayObject *l, *ldiag, *jl, *il, *r;
    int n, offset;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!i", &PyArray_Type, &l,
        &PyArray_Type, &ldiag, &PyArray_Type, &jl, &PyArray_Type, &il,
        &PyArray_Type, &r, &offset)) return NULL;

    n = ldiag->dimensions[0];

    inv_Lt_x_cd(n, (double *) l->data, (double *) ldiag->data,
        (int *) jl->data, (int *) il->data, (double *) r->data, offset);

    Py_INCREF(r);
    return (PyObject *) r;

}


THCTK_PRIVATE
void inv_LtL_x_cd(int n, const double *l, const double *ldiag, const int *jl,
    const int *il, double *r, int offset) {

    /* suffix _cd stands for column storage with separate diagonal */

    int j, m;

    if (offset != 0) {  /* Fortran convention, indices start at 1 */
        offset = 1;
        l--;
        ldiag--;
        il--;
        jl--;
        r--;
        m = n;
    } else {
        m = n - 1;
    }

    /* Solve L x = r and store the result in r. */

    for (j=offset; j<=m; j++) {
        register int k;
        register double temp = r[j]/ldiag[j];
        for (k=jl[j]; k<jl[j+1]; k++) r[il[k]] -= l[k]*temp;
        r[j] = temp;
    }

    /* Solve L' x = r and store the result in r. */

    r[m] /= ldiag[m];
    for (j=m-1; j>=offset; j--) {
        register int k;
        register double temp = 0;
        for (k=jl[j]; k<jl[j+1]; k++) temp += l[k]*r[il[k]];
        r[j] -= temp;
        r[j] /= ldiag[j];
    }
}


THCTKDOC(_numeric, inv_LtL_x_cd) =
"x = inv_LtL_x_cd(l, ldiag, jl, il, r, offset)\n\n"
"Compute x = A^-1 r for a (n x n) SPD-matrix with Cholesky decomposition\n"
"A = L^t L where the strict lower triangle of L is stored in l in compressed\n"
"column storage with row indices in il and column pointers in jl; the diagonal\n"
"of L is stored separately in ldiag. The vector r is overwritten by the\n"
"solution x on return.\n"
"If offset=0 the indices are in C convention (starting at 0) otherwise Fortran\n"
"convention (starting at 1) is assumed\n";

THCTKFUN(_numeric, inv_LtL_x_cd)
{

    PyArrayObject *l, *ldiag, *jl, *il, *r;
    int n, offset;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!i", &PyArray_Type, &l,
        &PyArray_Type, &ldiag, &PyArray_Type, &jl, &PyArray_Type, &il,
        &PyArray_Type, &r, &offset)) return NULL;

    n = ldiag->dimensions[0];

    inv_LtL_x_cd(n, (double *) l->data, (double *) ldiag->data,
        (int *) jl->data, (int *) il->data, (double *) r->data, offset);

    Py_INCREF(r);
    return (PyObject *) r;

}

THCTK_PRIVATE
void inv_LU_sq_x(int n, const double *l, const int *jl, const int *il,
                        const double *u, const int *ju, const int *iu,
                        double *r, int offset) {

    int j, m;
    
    
    if (offset != 0) {  /* Fortran convention, indices start at 1 */
        offset = 1;
        l--;
        il--;
        jl--;
        r--;
        m = n;
    } else {
        m = n - 1;
    }

    /* Solve L x = r and store the result in r. */

    for (j=offset; j<=m; j++) {
        register int k;
        register double temp = r[j];
        for (k=il[j]; k<il[j+1]-1; k++) {
          temp -= r[jl[k]]*l[k];
        }
        r[j] = temp/l[il[j+1]-1]; 
    }

    /* Solve U x = r and store the result in r. */
    
    for (j=m-1; j>=offset; j--) {
        register int k;
        register double temp = r[j];
        for (k=iu[j]+1; k<iu[j+1]; k++) {
          temp -= u[k]*r[ju[k]];
        }
        r[j] = temp; 
    }
}


THCTKDOC(_numeric, inv_LU_sq_x) = "";

THCTKFUN(_numeric, inv_LU_sq_x)
{

    PyArrayObject *l, *jl, *il, *u, *ju, *iu, *r;
    int n, offset;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!i", 
        &PyArray_Type, &l, &PyArray_Type, &jl, &PyArray_Type, &il,
        &PyArray_Type, &u, &PyArray_Type, &ju, &PyArray_Type, &iu,
        &PyArray_Type, &r, &offset)) return NULL;

    n = il->dimensions[0] - 1;

    inv_LU_sq_x(n, (double *) l->data, (int *) jl->data, (int *) il->data, 
                   (double *) u->data, (int *) ju->data, (int *) iu->data,
                   (double *) r->data, offset);

    Py_INCREF(r);
    return (PyObject *) r;

}



THCTKDOC(_numeric,poly_terms) = "";

THCTKFUN(_numeric,poly_terms)
{

    PyArrayObject *p, *ca, *xa, *ida, *t;
    double c, *x;
    char *m, *idx;
    int n, d, i, j, k;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!", &PyArray_Type, &p,
        &PyArray_Type, &ca, &PyArray_Type, &xa, &PyArray_Type, &ida,
        &PyArray_Type, &t)) return NULL;

    n = p->dimensions[0];
    if (n != ida->dimensions[0]) return NULL;
    d = p->dimensions[1];
    x = (double *) xa->data;

    for (i=0, m = p->data, idx = ida->data; i<n; i++,
            m += p->strides[0], idx += ida->strides[0]) {
        if ((c = ((double *) ca->data)[i])) {
            for (j=0; j<d; j++) {
                if ((k = ((int *) m)[j])) c *= pow(x[j], k);
            }
            *(double *) (t->data + (*((int *) idx) * t->strides[0])) += c;
        } else {
            *(double *) (t->data + (*((int *) idx) * t->strides[0])) = 0.0;
        }
    }

    Py_INCREF(t);
    return (PyObject *) t;

}


THCTKDOC(_numeric,poly_eval) = "";

THCTKFUN(_numeric,poly_eval)
{

    PyArrayObject *p, *ca, *xa;
    double c, *x, y = 0;
    char *m;
    int n, d, i, j, k;

    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &p,
        &PyArray_Type, &ca, &PyArray_Type, &xa)) return NULL;

    n = p->dimensions[0];
    d = p->dimensions[1];
    x = (double *) xa->data;

    for (i=0, m=p->data; i<n; i++, m+=p->strides[0]) {
        if ((c = ((double *) ca->data)[i])) {
            for (j=0; j<d; j++) {
                if ((k = ((int *) m)[j])) c *= pow(x[j], k);
            }
            y += c;
        }
    }

    return PyFloat_FromDouble(y);

}



THCTKDOC(_numeric,bosonelements) = "";

THCTKFUN(_numeric,bosonelements)
{
    PyArrayObject *Ad = NULL;
    double *d = NULL;
    int n, a, b, j, k, p, m, e = 0;

    /*
    for j in range(a, n):
        d[j] = 1
        for k in range(j-a+1,j+1):     d[j] *= k
        for k in range(j-a+1,j+b-a+1): d[j] *= k
    d = Numeric.sqrt(d[:n-abs(i)])
    */

    if (!PyArg_ParseTuple(args, "iii", &n, &p, &m))
        return NULL;

    e = n - abs(m - p);
    Ad = (PyArrayObject *) PyArray_FromDims(1, &e, PyArray_DOUBLE);
    d = (double *) Ad->data;
    a = min(p,m);
    b = max(p,m);

    for (j=0; j<e; j++) {
        d[j] = 1;
        for (k=j-a+1; k<=j; k++)        d[j] *= k;
        for (k=j-a+1; k<=j+b-a; k++)    d[j] *= k;
        d[j] = sqrt(d[j]);
    }

    return PyArray_Return(Ad);
}

THCTKDOC(_numeric,dp_index_dd) = "";

THCTKFUN(_numeric,dp_index_dd)
{

    PyArrayObject *A, *iindex, *jindex, *xa, *ya = NULL;
    double *x0, *y0, *x, *y = NULL; 
    double a, c = 0;
    int n, m, l, r, k, b, d, nel, iel, transpose = 0;
    int *i, *j = NULL;

    if (!PyArg_ParseTuple(args, "O!iiO!O!iiidO!O!", &PyArray_Type, &A,
        &n, &m, &PyArray_Type, &iindex, &PyArray_Type, &jindex,
        &transpose, &l, &r, &c, &PyArray_Type, &xa, &PyArray_Type, &ya))
        return NULL;

    nel = A->dimensions[0];

#ifndef NOthctkPyChecks
    if (A->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_IndexError,
            "The input matrix has the wrong typecode");
        return NULL;
    }
    if (xa->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_IndexError,
            "The input vector has the wrong typecode");
        return NULL;
    }
    if (ya->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_IndexError,
            "The output vector has the wrong typecode");
        return NULL;
    }
    if (A->nd != 1) {
        PyErr_SetString(PyExc_IndexError,
            "The input matrix has to be in Sparse Index format");
        return NULL;
    }
    if (!PyArray_ISCONTIGUOUS(iindex)) {
        PyErr_SetString(PyExc_IndexError,
            "The i-index array is not contiguous");
        return NULL;
    }
    if (!PyArray_ISCONTIGUOUS(jindex)) {
        PyErr_SetString(PyExc_IndexError,
            "The j-index array is not contiguous");
        return NULL;
    }
    if (!PyArray_ISCONTIGUOUS(xa)) {
        PyErr_SetString(PyExc_IndexError,
            "The input vector is not contiguous");
        return NULL;
    }
    if (!PyArray_ISCONTIGUOUS(ya)) {
        PyErr_SetString(PyExc_IndexError,
            "The output vector is not contiguous");
        return NULL;
    }
    if (xa->dimensions[0] != l*n*r) {
        PyErr_SetString(PyExc_IndexError,
            "The input vector has the wrong dimension");
        return NULL;
    }
    if (ya->dimensions[0] != l*n*r) {
        PyErr_SetString(PyExc_IndexError,
            "The output vector has the wrong dimension");
        return NULL;
    }
#endif // NOthctkPyChecks

    x0 = (double *) xa->data;
    y0 = (double *) ya->data;
    d = (n-1)*r;

    if (transpose) {
        i = (int *) jindex->data;
        j = (int *) iindex->data;
    } else {
        i = (int *) iindex->data;
        j = (int *) jindex->data;
    }

    for (iel=0; iel<nel; iel++, i++, j++) {
        a = c * (*(double *) (A->data + iel*A->strides[0]));
        if (a != 0) {
            x = x0 + (*j) * r;
            y = y0 + (*i) * r;
            for (b=0; b<l; b++, x+=d, y+=d) {
                for (k=0; k<r; k++, x++, y++) { *y += a * (*x); }
            }
        }
    }

    Py_INCREF(Py_None);
    return Py_None;

}


THCTKDOC(_numeric,dp_dense_dd) = "";

THCTKFUN(_numeric,dp_dense_dd)
{

    PyArrayObject *A, *xa, *ya = NULL;
    double *x0, *y0, *x, *y = NULL; 
    double a, c = 0;
    int n, l, r, i, j, k, b, d, transpose = 0;

    if (!PyArg_ParseTuple(args, "O!iiidO!O!", &PyArray_Type, &A, &transpose,
        &l, &r, &c, &PyArray_Type, &xa, &PyArray_Type, &ya))
        return NULL;

    n = A->dimensions[0];

#ifndef NOthctkPyChecks
    if (A->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_IndexError,
            "The input matrix has the wrong typecode");
        return NULL;
    }
    if (xa->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_IndexError,
            "The input vector has the wrong typecode");
        return NULL;
    }
    if (ya->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_IndexError,
            "The output vector has the wrong typecode");
        return NULL;
    }
    if (A->nd != 2) {
        PyErr_SetString(PyExc_IndexError,
            "The input matrix has to be two-dimensional");
        return NULL;
    }
    if (A->dimensions[1] != n) {
        PyErr_SetString(PyExc_IndexError,
            "The input matrix is not square");
        return NULL;
    }
    if (!PyArray_ISCONTIGUOUS(xa)) {
        PyErr_SetString(PyExc_IndexError,
            "The input vector is not contiguous");
        return NULL;
    }
    if (!PyArray_ISCONTIGUOUS(ya)) {
        PyErr_SetString(PyExc_IndexError,
            "The output vector is not contiguous");
        return NULL;
    }
    if (xa->dimensions[0] != l*n*r) {
        PyErr_SetString(PyExc_IndexError,
            "The input vector has the wrong dimension");
        return NULL;
    }
    if (ya->dimensions[0] != l*n*r) {
        PyErr_SetString(PyExc_IndexError,
            "The output vector has the wrong dimension");
        return NULL;
    }
#endif // NOthctkPyChecks

    x0 = (double *) xa->data;
    y0 = (double *) ya->data;
    d = (n-1)*r;

    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            if (transpose) a = c * (*(double *)
                (A->data + j*A->strides[0] + i*A->strides[1]));
            else a = c * (*(double *)
                (A->data + i*A->strides[0] + j*A->strides[1]));
            if (a != 0) {
                x = x0 + j*r;
                y = y0 + i*r;
                for (b=0; b<l; b++, x+=d, y+=d) {
                    for (k=0; k<r; k++, x++, y++) { *y += a * (*x); }
                }
            }
        }
    }

    Py_INCREF(Py_None);
    return Py_None;

}

#define EXDIFFMAX   3
THCTKDOC(_numeric,excitation_diff) =
"n, diff = excitation_diff(xa, xb, [nmax])\n"
"takes two integer arrays xa and xb of the same length representing\n"
"excitations and returns the number n of differences in the exitation\n"
"patterns as well as a tuple diff containing the indices of the positions\n"
"at which the differences occur.\n"
"The optional argument nmax (default: "  TOSTRING(EXDIFFMAX)
") determines how many differences are\n"
"maximally accounted for. The return value is (idx, None) if the differences\n"
"exceed nmax. idx is the index at which nmax was exceeded.\n"
;

THCTKFUN(_numeric,excitation_diff)
{

    PyArrayObject *xa, *xb;
    PyObject *diff;
    int n, nd, i, nmax = EXDIFFMAX;
    UINT8 *ia, *ib;

    if (!PyArg_ParseTuple(args, "O!O!|i",
        &PyArray_Type, &xa, &PyArray_Type, &xb, &nmax))
        return NULL;

    if (nmax < 0) nmax = EXDIFFMAX;
    n = xa->dimensions[0];
    nd = nmax;

#ifndef NOthctkPyChecks
    if (xa->nd != 1) {
        PyErr_SetString(PyExc_IndexError,
            "The first input array has to be a vector");
        return NULL;
    }
    if (xb->nd != 1) {
        PyErr_SetString(PyExc_IndexError,
            "The second input array has to be a vector");
        return NULL;
    }
    if (xa->descr->type_num != PyArray_UBYTE) {
        PyErr_SetString(PyExc_TypeError,
            "The first input vector has to be of typecode UnsignedInt8");
        return NULL;
    }
    if (xb->descr->type_num != PyArray_UBYTE) {
        PyErr_SetString(PyExc_TypeError,
            "The second input vector has to be of typecode UnsignedInt8");
        return NULL;
    }
    if (xb->dimensions[0] != n) {
        PyErr_SetString(PyExc_IndexError,
            "The input arrays must have the same length");
        return NULL;
    }
#endif // NOthctkPyChecks

    if (!(diff = PyTuple_New(nmax))) goto finish;
    ia = (UINT8 *) xa->data;
    ib = (UINT8 *) xb->data;
    nd = 0;
    for (i=0; i<n; i++, ia++, ib++) {
        if (*ia != *ib) {
            if (nd == nmax) {
                Py_DECREF(diff);
                Py_INCREF(Py_None);
                diff = Py_None;
                nd = i;
                goto finish;
            }
            PyTuple_SET_ITEM(diff, nd++, PyInt_FromLong(i));
        }
    }
    if (_PyTuple_Resize(&diff, nd)) nd = 0;

finish:
    return Py_BuildValue("(iN)", nd, diff);

}
#undef EXDIFFMAX

THCTKDOC(_numeric,exclist2array) =
"xv = exclist2array(idx, exc, xv)\n"
"takes two lists idx and exc of the same length which contain position\n"
"and excitation level, respectively. The values of the integer array xv\n"
"are set accordingly. The typecode of xv MUST be UnsignedInt8 .\n"
;

THCTKFUN(_numeric,exclist2array)
{

    PyArrayObject *xv;
    PyObject *idx, *exc;
    int n, xlen, i, j, pos=0, end;
    UINT8 *iv;

    if (!PyArg_ParseTuple(args, "O!O!O!",
        &PyList_Type, &idx, &PyList_Type, &exc, &PyArray_Type, &xv))
        return NULL;

    n = xv->dimensions[0];
    xlen = PyList_GET_SIZE(idx);

#ifndef NOthctkPyChecks
    if (xv->descr->type_num != PyArray_UBYTE) {
        PyErr_SetString(PyExc_TypeError,
            "The input array has to be of typecode UnsignedInt8");
        return NULL;
    }
    if (xv->nd != 1) {
        PyErr_SetString(PyExc_IndexError,
            "The input array has to be a vector");
        return NULL;
    }
    if (xlen != PyList_GET_SIZE(exc)) {
        PyErr_SetString(PyExc_IndexError,
            "The input lists have to be of the same length");
        return NULL;
    }
    if (n < PyList_GET_ITEM(idx, xlen)+1) {
        PyErr_SetString(PyExc_IndexError,
            "The input array is too short");
        return NULL;
    }
#endif // NOthctkPyChecks

    iv = (UINT8 *) xv->data;
    for (i=0; i<xlen; i++) {
        end = PyInt_AS_LONG(PyList_GET_ITEM(idx, i));
        if (end < pos) {
            PyErr_SetString(PyExc_IndexError,
                "The index list has to be sorted");
            return NULL;
        }
        for (j=pos; j<end; j++, iv++) *iv = 0;
        pos = end + 1;
        *iv++ = (UINT8) PyInt_AS_LONG(PyList_GET_ITEM(exc, i));
    }
    for (j=pos; j<n; j++, iv++) *iv = 0;

    Py_INCREF(xv);
    return (PyObject *) xv;
}

THCTKDOC(_numeric,VSCF_Scale_PairPotential) =
"\n"
;

THCTKFUN(_numeric,VSCF_Scale_PairPotential)
{

    PyArrayObject *Ax, *Ay, *A;
    double s, *Vx, *Vy;
    char *px;
    int nx, ny, sx, sy, i, j;

    if (!PyArg_ParseTuple(args, "dO!O!O!", &s, &PyArray_Type, &A,
        &PyArray_Type, &Ax, &PyArray_Type, &Ay))
        return NULL;

    nx = Ax->dimensions[0];
    ny = Ay->dimensions[0];
    sx = A->strides[0];
    sy = A->strides[1];
    Vx = (double *) Ax->data;
    Vy = (double *) Ay->data;
    px = A->data;
    for (i=0; i<nx; i++, px += sx) {
        double vx = Vx[i];
        char *V = px;
        for (j=0; j<ny; j++, V += sy) {
            double v = abs(*((double *) V));
            if (v > vx || v > Vy[j]) (*((double *) V)) *= s;
        }
    }
    Py_INCREF(Py_None);
    return Py_None;
}

THCTKDOC(_numeric,VSCF_Scale_TriplePotential) =
"\n"
;

THCTKFUN(_numeric,VSCF_Scale_TriplePotential)
{

    PyArrayObject *Ax, *Ay, *Az, *A;
    double s, *Vx, *Vy, *Vz;
    char *px;
    int nx, ny, nz, sx, sy, sz, i, j, k;

    if (!PyArg_ParseTuple(args, "dO!O!O!O!", &s, &PyArray_Type, &A,
        &PyArray_Type, &Ax, &PyArray_Type, &Ay, &PyArray_Type, &Az))
        return NULL;

    nx = Ax->dimensions[0];
    ny = Ay->dimensions[0];
    nz = Az->dimensions[0];
    sx = A->strides[0];
    sy = A->strides[1];
    sz = A->strides[2];
    Vx = (double *) Ax->data;
    Vy = (double *) Ay->data;
    Vz = (double *) Az->data;
    px = A->data;
    for (i=0; i<nx; i++, px += sx) {
        double vx = Vx[i];
        char *py = px;
        for (j=0; j<ny; j++, py += sy) {
            double vy = Vy[j];
            char *V = py;
            for (k=0; k<nz; k++, V += sz) {
                double v = abs(*((double *) V));
                if (v > vx || v > vy || v > Vz[k] ) (*((double *) V)) *= s;
            }
        }
    }
    Py_INCREF(Py_None);
    return Py_None;
}

THCTKDOC(_numeric, tensorOperatorIndexPair) =
"Function tensorOperatorIndexPaar computes indeces of a tensor product \n\
matrices, the product of which gives the wanted tensor product element\n\
Iput parameters:\n\
tuple of indices of a tensor matrix - indexI, indexJ\n\
tuple of integer arrays with indices of a single matrices, where results \
will be written in - i, j\n\
tuple of integer arrays with dimensions  of a single matrices - n, m.\n"
;

THCTKFUN(_numeric, tensorOperatorIndexPair)
{
    PyArrayObject *i, *j, *n, *m;
    int *ii, *jj, *nn, *mm, *tii=NULL, *tjj=NULL;
    size_t ndim;
    int indexI, indexJ, kk;

    if (!PyArg_ParseTuple(args, "(ii)(O!O!)(O!O!)", &indexI, &indexJ, 
        &PyArray_Type, &i, &PyArray_Type, &j,
        &PyArray_Type, &n, &PyArray_Type, &m))
        return NULL;

    ii = (int *) i->data;
    jj = (int *) j->data;
    nn = (int *) n->data;
    mm = (int *) m->data;

    ndim = i->dimensions[0];
    if ( (ndim!=j->dimensions[0]) || (ndim!=n->dimensions[0]) || (ndim!=m->dimensions[0])) {
        PyErr_SetString(PyExc_IndexError,
            "i, j, n, m must have the same length!\n");
        return NULL;
        }

    if (ndim > 0) {
        if ((tii = (int *) malloc((ndim+1)*sizeof(int))) == NULL) goto fail_malloc;
        if ((tjj = (int *) malloc((ndim+1)*sizeof(int))) == NULL) goto fail_malloc;
    }
    
    tii[ndim] = indexI;
    tjj[ndim] = indexJ;
    kk=ndim;
    while (kk>0){
        kk--;
        size_t dimI=nn[kk], dimJ=mm[kk];
        ii[kk] = tii[kk+1] % dimI;
        jj[kk] = tjj[kk+1] % dimJ;
        tii[kk] = (tii[kk+1] - ii[kk]) / dimI;
        tjj[kk] = (tjj[kk+1] - jj[kk]) / dimJ;
    }

fail_malloc:
    if (tjj != NULL) free((void *) tjj);
    if (tii != NULL) free((void *) tii);

    Py_INCREF(Py_None);
    return Py_None;
}

THCTKDOC(_numeric, productOperatorSingle2CSR) =
"Function productOperatorSingle2CSR computes the elements of a tensor product\n\
of identity matrices with 1 non-identity matrix.\n\
Iput parameters:\n\
tensor product in CSR format - CSRi, CSRj, CSRd\n\
numbers of non-identity matrix - k (must be int)\n\
non-identity matrix - ak\n\
tuple with dimensions of all matrices of the tensor product - TUP\n\
offset is set to 0.\n"
;

THCTKFUN(_numeric, productOperatorSingle2CSR)
{
    PyArrayObject *CSRi, *CSRj, *CSRd, *Ak;
    PyObject *TUP;
    size_t ndim, *dims=NULL;
    double *cd, *ak;
    int *ci, *cj;
    int k, lci, offset=0, i, j, *tmpi=NULL, *tmpj=NULL, *ii=NULL, *jj=NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!iO!O!|i", &PyArray_Type, &CSRi,
        &PyArray_Type, &CSRj, &PyArray_Type, &CSRd, &k,
        &PyArray_Type, &Ak, &PyTuple_Type, &TUP, &offset))
        return NULL;

    if (!PyTuple_Check(TUP)) {
        return NULL;
        }
    if (!PyArray_Check(Ak)){
        return NULL;
        }
    if (!PyArray_ISCONTIGUOUS(Ak)) {
        PyErr_SetString(PyExc_MemoryError, "Ak is not contiguous");
        return NULL;
    }
    if (Ak->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_IndexError,
            "Ak matrix has the wrong typecode");
        return NULL;
    }

    ndim = PyTuple_GET_SIZE(TUP); 
    if (ndim > 0) {
        if ((dims = (size_t *) malloc(ndim*sizeof(size_t))) == NULL) goto fail_malloc;
        if ((tmpi = (int *) malloc((ndim+1)*sizeof(int))) == NULL) goto fail_malloc;
        if ((tmpj = (int *) malloc((ndim+1)*sizeof(int))) == NULL) goto fail_malloc;
        if ((ii = (int *) malloc(ndim*sizeof(int))) == NULL) goto fail_malloc;
        if ((jj = (int *) malloc(ndim*sizeof(int))) == NULL) goto fail_malloc;
    }
    for (i=0; i<ndim; i++) {
        dims[i] = PyInt_AS_LONG(PyTuple_GetItem(TUP, i));
    }
    k -= offset;

    ak = (double *) Ak->data;
    ci = (int *) CSRi->data;
    cj = (int *) CSRj->data;
    cd = (double *) CSRd->data;
    lci = CSRi->dimensions[0];

    for (i=0; i<lci-1; i++){
        for (j=ci[i]; j<ci[i+1]; j++){
            int kk=ndim;
            cd[j] = 0;
            tmpi[ndim] = i;
            tmpj[ndim] = cj[j];
            while (kk>0) {
                kk--;
                size_t dim = dims[kk];
                ii[kk] = tmpi[kk+1] % dim; 
                jj[kk] = tmpj[kk+1] % dim; 
                tmpi[kk] = (tmpi[kk+1] - ii[kk]) / dim;
                tmpj[kk] = (tmpj[kk+1] - jj[kk]) / dim;
                if ((kk!=k) && (ii[kk]!=jj[kk])) {
                    goto zero_element;
                }
            }
            cd[j] = ak[ii[k]*dims[k]+jj[k]];
            zero_element:{
            }
        }
    }

fail_malloc:
    if (jj != NULL) free((void *) jj);
    if (ii != NULL) free((void *) ii);
    if (tmpj != NULL) free((void *) tmpj);
    if (tmpi != NULL) free((void *) tmpi);
    if (dims != NULL) free((void *) dims);

    Py_INCREF(Py_None);
    return Py_None;
}

THCTKDOC(_numeric, productOperatorDouble2CSR) =
"Function productOperatorDouble2CSR computes the elements of a tensor product\n\
of identity matrices with 2 non-identity matrices.\n\
Iput parameters:\n\
tensor product in CSR format - CSRi, CSRj, CSRd\n\
tuple of length 2 with numbers of non-identity matrices - k,l\n\
tuple of length 2 with non-identity matrices - ak, bk\n\
tuple with dimensions of all matrices of the tensor product - TUP\n\
offset is set to 0.\n"
;

THCTKFUN(_numeric, productOperatorDouble2CSR)
{
    PyArrayObject *CSRi, *CSRj, *CSRd, *Ak, *Bk;
    PyObject *TUP;
    size_t ndim, *dims=NULL;
    double *cd, *ak, *bk;
    int k, l, lci, offset=0, i, j, *tmpi=NULL, *tmpj=NULL, *ii=NULL, *jj=NULL;
    int *ci, *cj;

    if (!PyArg_ParseTuple(args, "O!O!O!(ii)(O!O!)O!|i", &PyArray_Type, &CSRi,
        &PyArray_Type, &CSRj, &PyArray_Type, &CSRd, &k, &l,
        &PyArray_Type, &Ak, &PyArray_Type, &Bk, 
        &PyTuple_Type, &TUP, &offset))
        return NULL;

    if (!PyTuple_Check(TUP)) {
        return NULL;
        }
    if (!PyArray_Check(Ak)){
        return NULL;
        }
    if (!PyArray_ISCONTIGUOUS(Ak)) {
        PyErr_SetString(PyExc_MemoryError, "Ak is not contiguous");
        return NULL;
        }
    if (Ak->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_IndexError,
            "Ak matrix has the wrong typecode");
        return NULL;
    }
    if (!PyArray_Check(Bk)){
        return NULL;
        }
    if (!PyArray_ISCONTIGUOUS(Bk)) {
        PyErr_SetString(PyExc_MemoryError, "Bk is not contiguous");
        return NULL;
        }
    if (Bk->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_IndexError,
            "Bk matrix has the wrong typecode");
        return NULL;
    }

    ndim = PyTuple_GET_SIZE(TUP); 
    if (ndim > 0) {
        if ((dims = (size_t *) malloc(ndim*sizeof(size_t))) == NULL) goto fail_malloc;
        if ((tmpi = (int *) malloc((ndim+1)*sizeof(int))) == NULL) goto fail_malloc;
        if ((tmpj = (int *) malloc((ndim+1)*sizeof(int))) == NULL) goto fail_malloc;
        if ((ii = (int *) malloc(ndim*sizeof(int))) == NULL) goto fail_malloc;
        if ((jj = (int *) malloc(ndim*sizeof(int))) == NULL) goto fail_malloc;
    }
    for (i=0; i<ndim; i++) {
        dims[i] = PyInt_AS_LONG(PyTuple_GetItem(TUP, i));
    }
    k -= offset;
    l -= offset;

    ak = (double *) Ak->data;
    bk = (double *) Bk->data;
    ci = (int *) CSRi->data;
    cj = (int *) CSRj->data;
    cd = (double *) CSRd->data;
    lci = CSRi->dimensions[0];

    for (i=0; i<lci-1; i++){
        for (j=ci[i]; j<ci[i+1]; j++){
            long kk=ndim;
            cd[j] = 0;
            tmpi[ndim] = i;
            tmpj[ndim] = cj[j];
            while (kk>0) {
                kk--;
                size_t dim = dims[kk];
                ii[kk] = tmpi[kk+1] % dim; 
                jj[kk] = tmpj[kk+1] % dim; 
                tmpi[kk] = (tmpi[kk+1] - ii[kk]) / dim;
                tmpj[kk] = (tmpj[kk+1] - jj[kk]) / dim;
                if ((ii[kk]!=jj[kk]) && (kk!=k) && (kk!=l)) {
                    goto zero_element;
                }
            }
            cd[j] = ak[ii[k]*dims[k]+jj[k]] * bk[ii[l]*dims[l]+jj[l]] ;
            zero_element:{
            }
        }
    }

fail_malloc:
    if (jj != NULL) free((void *) jj);
    if (ii != NULL) free((void *) ii);
    if (tmpj != NULL) free((void *) tmpj);
    if (tmpi != NULL) free((void *) tmpi);
    if (dims != NULL) free((void *) dims);

    Py_INCREF(Py_None);
    return Py_None;
}

THCTKDOC(_numeric, productOperatorN2CSR) =
"Function productOperatorN2CSR computes the elements of a tensor product\n\
from smaller matrices. The number of non-identity matrices is arbitrary.\n\
Iput parameters:\n\
tensor product in CSR format - CSRi, CSRj, CSRd\n\
tuple with numbers of non-identity matrices - INDEX\n\
tuple with non-identity matrices - OPERATORS (must have same length as INDEX)\n\
tuple with dimensions of all matrices of the tensor product - TUP\n\
offset is set to 0.\n"
;

THCTKFUN(_numeric, productOperatorN2CSR)
{
    PyArrayObject *CSRi, *CSRj, *CSRd;
    PyObject *TUP, *INDEX, *OPERATORS;
    size_t ndim, lenIN, lenOP;
    size_t *dims=NULL, *index=NULL;
    double **matrices=NULL, *cd;
    int i, j, lci, offset=0, *tmpi=NULL, *tmpj=NULL, *ii=NULL, *jj=NULL;
    int *ci, *cj, failure=0;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!|i", &PyArray_Type, &CSRi,
        &PyArray_Type, &CSRj, &PyArray_Type, &CSRd,
        &PyTuple_Type, &INDEX, &PyTuple_Type, &OPERATORS, &PyTuple_Type, &TUP,
        &offset))
        return NULL;

    if (!PyTuple_Check(INDEX)) {
        PyErr_SetString(PyExc_TypeError,
            "INDEX must be of tuple type!\n");
        return NULL;
        }
    if (!PyTuple_Check(OPERATORS)) {
        PyErr_SetString(PyExc_TypeError,
            "OPERATORS must be of tuple type!\n");
        return NULL;
        }
    if (!PyTuple_Check(TUP)) {
        PyErr_SetString(PyExc_TypeError,
            "TUP must be of tuple type!\n");
        return NULL;
        }

    ndim = PyTuple_GET_SIZE(TUP); 
    lenIN = PyTuple_GET_SIZE(INDEX); 
    lenOP = PyTuple_GET_SIZE(OPERATORS); 

    if (lenIN!=lenOP){
        PyErr_SetString(PyExc_IndexError,
            "the length of OPERATORS and INDEX tuples must be equal!\n");
        return NULL;
        }

    if (ndim > 0) {
        if ((dims = (size_t *) malloc(ndim*sizeof(size_t))) == NULL) goto fail_malloc;
        if ((index = (size_t *) malloc(lenIN*sizeof(size_t))) == NULL) goto fail_malloc;
        if ((matrices = (double **) malloc(ndim*sizeof(double *))) == NULL) goto fail_malloc;
        if ((tmpi = (int *) malloc((ndim+1)*sizeof(int))) == NULL) goto fail_malloc;
        if ((tmpj = (int *) malloc((ndim+1)*sizeof(int))) == NULL) goto fail_malloc;
        if ((ii = (int *) malloc(ndim*sizeof(int))) == NULL) goto fail_malloc;
        if ((jj = (int *) malloc(ndim*sizeof(int))) == NULL) goto fail_malloc;
    }

    for (i=0; i<ndim; i++) {
        dims[i] = PyInt_AS_LONG(PyTuple_GetItem(TUP, i));
        matrices[i] = NULL;
        }

    for (i=0; i<lenOP; i++) {
        PyArrayObject *MAT = (PyArrayObject *) PyTuple_GetItem(OPERATORS,i);
        matrices[ PyInt_AS_LONG(PyTuple_GetItem(INDEX, i)) ] = (double *) MAT->data;
        if (!PyArray_ISCONTIGUOUS(MAT)) {
            PyErr_SetString(PyExc_MemoryError, "one of the non-identity matrices is not contiguous");
            failure=1;
            goto fail_malloc;
        }
        if (MAT->descr->type_num != PyArray_DOUBLE) {
            PyErr_SetString(PyExc_IndexError,
                "one of the non-identity matrices has the wrong typecode");
            failure=2;
            goto fail_malloc;
        }
    }

    ci = (int *) CSRi->data;
    cj = (int *) CSRj->data;
    cd = (double *) CSRd->data;
    lci = CSRi->dimensions[0];

    for (i=0; i<lci-1; i++){
        for (j=ci[i]; j<ci[i+1]; j++){
            long kk=ndim;
            cd[j] = 1;
            tmpi[ndim] = i;
            tmpj[ndim] = cj[j];
            while (kk>0) {
                kk--;
                size_t dim = dims[kk];
                ii[kk] = tmpi[kk+1] % dim; 
                jj[kk] = tmpj[kk+1] % dim; 
                tmpi[kk] = (tmpi[kk+1] - ii[kk]) / dim;
                tmpj[kk] = (tmpj[kk+1] - jj[kk]) / dim;
                if (matrices[kk]==NULL) {
                    if (ii[kk]!=jj[kk]) {
                        cd[j] = 0;
                        goto zero_element;
                    } 
                } else {
                    cd[j] *= matrices[kk][ii[kk]*dims[kk]+jj[kk]];
                }
            }
            zero_element:{
            }
        }
    }

fail_malloc:
    if (jj != NULL) free((void *) jj);
    if (ii != NULL) free((void *) ii);
    if (tmpj != NULL) free((void *) tmpj);
    if (tmpi != NULL) free((void *) tmpi);
    if (matrices != NULL) free((void *) matrices);
    if (index != NULL) free((void *) index);
    if (dims != NULL) free((void *) dims);

    if (failure > 0) {
        return NULL;
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

THCTK_PRIVATE
int chebyshevPropagationStep(
    const double *m, 
    const double *d, 
    const int *mj, 
    const int *mi, 
    double *xx, 
    double *yy, 
    double *bessel, 
    double *t1, 
    double *t2, 
    int order, 
    int steps, 
    double exp, 
    int n, 
    int offset  
    ) {
    
    int i=0, j=0, k=0;

    for (i=0; i<steps; i++) {
        amux_CSRd(m, d, mj, mi, xx, t1, n, offset);
        for (j=0; j<n; j++) {
            t1[j] *= -1;
            yy[j] = bessel[0] * xx[j] + bessel[1] * t1[j];
            } 
        for (j=2; j<order; j++) {
            printf("%i ", j);
            printf(" %f %f", t1[0], t1[1]);
            amux_CSRd(m, d, mj, mi, t1, t2, n, offset);
            for (k=0; k<n; k++) {
                t2[k] *= -2;
                t2[k] += t1[k];
                yy[k] += t2[k] * bessel[j];
                }
            printf("\t%f %f\n", t2[0], t2[1]);
            t1 = t2;
            } 
        for (j=0; j<n; j++) yy[j] *= exp;
            xx = yy;
        }
    return 1;
}

THCTK_PRIVATE
int chebyshevPropagationStep_re_complex(
    const double *m, 
    const double *d, 
    const int *mj, 
    const int *mi, 
    double *xx, 
    double *yy, 
    double *bessel, 
    double *t1, 
    double *t2, 
    int order, 
    int steps, 
    double exp_r, 
    double exp_i, 
    int n, 
    int offset  
    ) {
    
    int i=0, j=0, k=0;
    double tmp_r=0, tmp_i=0;
    double *temp;

    for (i=0; i<steps; i++) {
//      printf("%i ", i);
//      printf(" %f %f", xx[0], xx[1]);
        amux_CSRd_re_complex(m, d, mj, mi, xx, t1, n, offset);
        for (j=0; j<n; j++) {
            t1[2*j]   *= -1;
            t1[2*j+1] *=  1;
            yy[2*j]   = bessel[0] * xx[2*j]   + bessel[1] * t1[2*j+1];
            yy[2*j+1] = bessel[0] * xx[2*j+1] + bessel[1] * t1[2*j];
            } 
        for (j=2; j<order; j++) {
            printf("%i ", j);
            printf(" %f %f", xx[0], t1[0]);
            amux_CSRd_re_complex(m, d, mj, mi, t1, t2, n, offset);
            for (k=0; k<n; k++) {
                t2[2*k]   *= -2;
                t2[2*k+1] *=  2;
                t2[2*k+1] += xx[2*k];
                t2[2*k]   += xx[2*k+1];
                yy[2*k]   += t2[2*k+1] * bessel[j];
                yy[2*k+1] += t2[2*k]   * bessel[j];
                }
            printf("\t%f %f\n", t1[0], t2[0]);
            xx = t1;
            t1 = t2;
            } 
        for (j=0; j<n; j++) {
            tmp_r = yy[2*j]*exp_r - yy[2*j+1]*exp_i;
            tmp_i = yy[2*j]*exp_i + yy[2*j+1]*exp_r;
            yy[2*j]    = tmp_r;
            yy[2*j+1]  = tmp_i;
            }
//          printf(" %f %f\n", yy[0], yy[1]);
            xx = yy;
        }
    return 1;
}

THCTK_PRIVATE
int chebyshevPropagationStep_complex(
    const double *m, 
    const double *d, 
    const int *mj, 
    const int *mi, 
    double *xx, 
    double *yy, 
    double *bessel, 
    double *t1, 
    double *t2, 
    int order, 
    int steps, 
    double exp_r, 
    double exp_i, 
    int n, 
    int offset  
    ) {
    
    int i=0, j=0, k=0;
    double tmp_r=0, tmp_i=0;

    for (i=0; i<steps; i++) {
        amux_CSRd_complex(m, d, mj, mi, xx, t1, n, offset);
        for (j=0; j<n; j++) {
            t1[2*j]   *= -1;
            t1[2*j+1] *=  1;
            yy[2*j]   = bessel[0] * xx[2*j]   + bessel[1] * t1[2*j+1];
            yy[2*j+1] = bessel[0] * xx[2*j+1] + bessel[1] * t1[2*j];
            } 
        for (j=2; j<order; j++) {
            amux_CSRd_complex(m, d, mj, mi, t1, t2, n, offset);
            for (k=0; k<n; k++) {
                t2[2*k]   *= -2;
                t2[2*k+1] *=  2;
                t2[2*k+1] += xx[2*k];
                t2[2*k]   += xx[2*k+1];
                yy[2*k]   += t2[2*k+1] * bessel[j];
                yy[2*k+1] += t2[2*k]   * bessel[j];
                }
            t1 = t2;
            } 
        for (j=0; j<n; j++) {
            tmp_r = yy[2*j]*exp_r - yy[2*j+1]*exp_i;
            tmp_i = yy[2*j]*exp_i + yy[2*j+1]*exp_r;
            yy[2*j]    = tmp_r;
            yy[2*j+1]  = tmp_i;
            }
            xx = yy;
        }
    return 1;
}


THCTKDOC(_numeric, chebyshevPropagationStep) =
"smth\n";

THCTKFUN(_numeric, chebyshevPropagationStep)
{
    PyArrayObject *a, *ad, *aj, *ai, *x, *y, *bessel, *p=NULL;
    int n, order, steps=1, offset=0;
    double exp, *t1, *t2, *yy;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!iid|iO!", 
        &PyArray_Type, &a,
        &PyArray_Type, &ad, 
        &PyArray_Type, &aj, 
        &PyArray_Type, &ai,
        &PyArray_Type, &x, 
        &PyArray_Type, &y, 
        &PyArray_Type, &bessel, 
        &order, &steps, &exp,
        &offset, 
        &PyArray_Type, &p))
        return NULL;

    n = ai->dimensions[0] - 1 + offset;
    yy = ((double *) y->data) - offset;
    if (n > 0) {
        if ((t1 = (double *) malloc(n*sizeof(double))) == NULL) goto fail_malloc;
        if ((t2 = (double *) malloc(n*sizeof(double))) == NULL) goto fail_malloc;
        }

    if (! chebyshevPropagationStep(((double *) a->data) - offset,
                        ((double *) ad->data) - offset,
                        ((int *) aj->data) - offset,
                        ((int *) ai->data) - offset,
                        ((double *) x->data) - offset,
                        (double *) yy,
                        ((double *) bessel->data),
                        (double *) t1,
                        (double *) t2,
                        order, steps, exp, n, offset
                        )) return NULL;

fail_malloc:
    if (t1 != NULL) free((void *) t1);
    if (t2 != NULL) free((void *) t2);

    Py_INCREF(Py_None);
    return Py_None;

}

THCTKDOC(_numeric, chebyshevPropagationStep_re_complex) =
"smth\n";

THCTKFUN(_numeric, chebyshevPropagationStep_re_complex)
{
    PyArrayObject *a, *ad, *aj, *ai, *x, *y, *bessel, *p=NULL;
    int n, order, steps=1, offset=0;
    double exp_r, exp_i, *t1, *t2;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!iid|diO!", 
        &PyArray_Type, &a,
        &PyArray_Type, &ad, 
        &PyArray_Type, &aj, 
        &PyArray_Type, &ai,
        &PyArray_Type, &x, 
        &PyArray_Type, &y, 
        &PyArray_Type, &bessel, 
        &order, &steps, &exp_r,
        &exp_i,
        &offset, 
        &PyArray_Type, &p))
        return NULL;

    n = ai->dimensions[0] - 1 + offset;
    if (n > 0) {
        if ((t1 = (double *) malloc(2*n*sizeof(double))) == NULL) goto fail_malloc;
        if ((t2 = (double *) malloc(2*n*sizeof(double))) == NULL) goto fail_malloc;
        }

    if (! chebyshevPropagationStep_re_complex(((double *) a->data) - offset,
                        ((double *) ad->data) - offset,
                        ((int *) aj->data) - offset,
                        ((int *) ai->data) - offset,
                        ((double *) x->data) - offset,
                        ((double *) y->data) - offset,
                        ((double *) bessel->data),
                        (double *) t1,
                        (double *) t2,
                        order, steps, exp_r, exp_i, n, offset
                        )) return NULL;

fail_malloc:
    if (t1 != NULL) free((void *) t1);
    if (t2 != NULL) free((void *) t2);

    Py_INCREF(Py_None);
    return Py_None;

}


THCTKDOC(_numeric, chebyshevPropagationStep_complex) =
"smth\n";

THCTKFUN(_numeric, chebyshevPropagationStep_complex)
{
    PyArrayObject *a, *ad, *aj, *ai, *x, *y, *bessel, *p=NULL;
    int n, order, steps=1, offset=0;
    double exp_r, exp_i, *t1, *t2;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!iid|diO!", 
        &PyArray_Type, &a,
        &PyArray_Type, &ad, 
        &PyArray_Type, &aj, 
        &PyArray_Type, &ai,
        &PyArray_Type, &x, 
        &PyArray_Type, &y, 
        &PyArray_Type, &bessel, 
        &order, &steps, &exp_r,
        &exp_i,
        &offset, 
        &PyArray_Type, &p))
        return NULL;

    n = ai->dimensions[0] - 1 + offset;
    if (n > 0) {
        if ((t1 = (double *) malloc(2*n*sizeof(double))) == NULL) goto fail_malloc;
        if ((t2 = (double *) malloc(2*n*sizeof(double))) == NULL) goto fail_malloc;
        }

    if (! chebyshevPropagationStep_complex(((double *) a->data) - offset,
                        ((double *) ad->data) - offset,
                        ((int *) aj->data) - offset,
                        ((int *) ai->data) - offset,
                        ((double *) x->data) - offset,
                        ((double *) y->data) - offset,
                        ((double *) bessel->data),
                        (double *) t1,
                        (double *) t2,
                        order, steps, exp_r, exp_i, n, offset
                        )) return NULL;

fail_malloc:
    if (t1 != NULL) free((void *) t1);
    if (t2 != NULL) free((void *) t2);

    Py_INCREF(Py_None);
    return Py_None;

}

#if defined(PY_VERSION_HEX) && PY_VERSION_HEX >= 0x02040000
    /* everything that needs Python 2.4 or higher, e.g. sets, goes here */
#endif

/*  here follows the module initialization
 *
 *  you have to add one line like
 *
 *    THCTKDEF(module,function)
 *
 *  for each function that you have defined above
 */

static struct PyMethodDef _numeric_methods[] = {
    THCTKDEF(_numeric, inv_L_x_cd)
    THCTKDEF(_numeric, inv_Lt_x_cd)
    THCTKDEF(_numeric, inv_LtL_x_cd)
    THCTKDEF(_numeric, inv_LU_sq_x)
    THCTKDEF(_numeric, amux_CSR)
    THCTKDEF(_numeric, amux_CSR_complex)
    THCTKDEF(_numeric, amux_CSR_re_complex)
    THCTKDEF(_numeric, amux_CSRd)
    THCTKDEF(_numeric, amux_CSRd_complex)
    THCTKDEF(_numeric, amux_CSRd_re_complex)
    THCTKDEF(_numeric, daxpy_p)
    THCTKDEF(_numeric, colamd)
    THCTKDEF(_numeric, copyArray)
    THCTKDEF(_numeric, dp_dense_dd)
    THCTKDEF(_numeric, dp_index_dd)
    THCTKDEF(_numeric, bosonelements)
    THCTKDEF(_numeric, poly_eval)
    THCTKDEF(_numeric, poly_terms)
    THCTKDEF(_numeric, excitation_diff)
    THCTKDEF(_numeric, exclist2array)
    THCTKDEF(_numeric, VSCF_Scale_PairPotential)
    THCTKDEF(_numeric, VSCF_Scale_TriplePotential)
    THCTKDEF(_numeric, productOperatorSingle2CSR)
    THCTKDEF(_numeric, productOperatorDouble2CSR)
    THCTKDEF(_numeric, productOperatorN2CSR)
    THCTKDEF(_numeric, tensorOperatorIndexPair)
    THCTKDEF(_numeric, chebyshevPropagationStep)
    THCTKDEF(_numeric, chebyshevPropagationStep_complex)
    THCTKDEF(_numeric, chebyshevPropagationStep_re_complex)
    {NULL, NULL, 0, NULL}
};

static char _numeric_module_documentation[] = "";

THCTKMOD(_numeric)
#endif
