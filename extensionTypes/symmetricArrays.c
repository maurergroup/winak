/*
 * thctk.extensionTypes.symmetricArrays symmetricArrays.c
 *
 *   thctk - python package for Theoretical Chemistry
 *   Copyright (C) 2006 Christoph Scheurer
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

#include <math.h>
#include <Python.h>
#include "structmember.h"

/*  For usage of the THCTK macros look in numeric/thctk_numeric.c */
#include "thctk.h"

#define SYMARRAY_MAXDIM 4

typedef struct {
    PyObject_HEAD
    PyArrayObject *array;
    PyObject *dimensions;
    char *data;
    int dim;
    int nel;
    int stride;
    int ndim[SYMARRAY_MAXDIM];
    PyArrayObject *idx[SYMARRAY_MAXDIM];
} symArray;

static void
symArray_dealloc(symArray* self)
{
    int i;

    Py_XDECREF(self->array);
    Py_XDECREF(self->dimensions);
    for (i=0; i < self->dim; i++) {
        Py_XDECREF((PyObject *) self->idx[i]);
    }
    self->ob_type->tp_free((PyObject *) self);
}

static PyObject *
symArray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    symArray *self;
    int i;

    self = (symArray *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->array = NULL;
        self->dimensions = NULL;
        self->data = NULL;
        self->dim = 0;
        self->stride = 0;
        for (i=0; i < SYMARRAY_MAXDIM; i++) {
            self->ndim[i] = 0;
            self->idx[i] = NULL;
        }
    }

    return (PyObject *)self;
}

/* forward definition */
static int symArray_setarray(symArray *self, PyObject *array, void *);

static int
symArray_init(symArray *self, PyObject *args, PyObject *kwds)
{
    PyObject *dimensions=NULL, *array=NULL;

    static char *kwlist[] = {"dimensions", "array", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, 
                                      &dimensions, &array))
        return -1; 

    if (dimensions) {
        Py_ssize_t pos;
        int nel=1;
        if (!PyTuple_Check(dimensions)) {
            PyErr_SetString(PyExc_TypeError, "dimensions has to be a tuple");
            return -1;
        }
        self->dim = PyTuple_Size(dimensions);
        if (self->dim < 1 || self->dim > SYMARRAY_MAXDIM) {
            PyErr_SetString(PyExc_NotImplementedError,
                "0 < dim < " TOSTRING(SYMARRAY_MAXDIM));
            return -1;
        }
        for (pos=0; pos < self->dim; pos++) {
            long i;
            PyObject* io = PyTuple_GetItem(dimensions, pos);
            if (!io) return -1;
            if ((i = PyInt_AsLong(io)) == -1) {
                if (PyErr_Occurred()) return -1;
            }
            self->ndim[pos] = i;
            nel *= (i*(i+1))/2;
        }
        self->nel = nel;
        Py_INCREF(dimensions);
        self->dimensions = dimensions;
    }

    if (array) {    // too simplistic
        if (symArray_setarray(self, array, NULL) == -1) return -1;
    }

    return 0;
}

static PyMemberDef symArray_members[] = {
    {"dim", T_INT, offsetof(symArray, dim), 0, "dimensionality"},
    {"dimensions", T_OBJECT, offsetof(symArray, dimensions), 0, "dimensions"},
    {"nel", T_INT, offsetof(symArray, nel), 0, "number of elements"},
    {NULL}  /* Sentinel */
};

static PyObject *
symArray_getarray(symArray *self, void *closure)
{
    if (! self->array || ! PyArray_Check(self->array)) {
        PyErr_SetString(PyExc_AttributeError, 
                    "The array attribute is not set");
        return NULL;
    }
    Py_INCREF(self->array);
    return (PyObject *) self->array;
}

static int
symArray_setarray(symArray *self, PyObject *value, void *closure)
{
    PyArrayObject *A = NULL;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the array attribute");
        return -1;
    }
  
    if (! PyArray_Check(value)) {
        PyErr_SetString(PyExc_TypeError, 
                    "The array attribute value must be a Numeric array");
        return -1;
    }
    A = (PyArrayObject *) value;

    if (A->nd > 1 ){
        PyErr_SetString(PyExc_IndexError, 
                    "The array has to be 1-D");
        return -1;
    }
    if (! PyArray_ISCONTIGUOUS(A)) {
        PyErr_SetString(PyExc_TypeError, 
                    "The array attribute value must be contiguous");
        return -1;
    }
    if (A->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, 
                    "The array attribute value must be PyArray_DOUBLE");
        return -1;
    }
    if (PyArray_SIZE(A) != self->nel) {
        PyErr_SetString(PyExc_IndexError, 
                    "The array has the wrong length");
        return -1;
    }
      
    Py_XDECREF(self->array);
    Py_INCREF(A);
    self->array = A;
    self->data = A->data;
    self->stride = A->strides[0];

    return 0;
}

static PyGetSetDef symArray_getseters[] = {
    {"array", 
     (getter)symArray_getarray, (setter)symArray_setarray,
     "double precision data array",
     NULL},
    {NULL}  /* Sentinel */
};

static PyObject *
symArray_fullContract(symArray* self, PyObject *args)
{
    int k, k2 = 0;
    Py_ssize_t i, j, a, b, offset=0;
    int *d = self->ndim;

    /* no error checking in this loop! */
    for (k=0; k < self->dim; k++, k2 +=2, d++) {
        a = PyInt_AsLong(PyTuple_GET_ITEM(args, k2));
        b = PyInt_AsLong(PyTuple_GET_ITEM(args, k2+1));
        i = max(a, b);
        j = min(a, b);
        offset *= ((*d) * ((*d) + 1))/2;
        offset += (i * (i + 1))/2 + j;
    }

    if (offset >= self->nel ){
        PyErr_SetString(PyExc_IndexError, "index too large");
        return NULL;
    }

    offset *= self->stride;
    return PyFloat_FromDouble(*((double *) (self->data + offset)));
}

static PyMethodDef symArray_methods[] = {
    {"fullContract", (PyCFunction)symArray_fullContract, METH_VARARGS,
     ""
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject symArrayType = {
    PyObject_HEAD_INIT(NULL)
    0,                          /*ob_size*/
    "symmetricArrays.symArray", /*tp_name*/
    sizeof(symArray),           /*tp_basicsize*/
    0,                          /*tp_itemsize*/
    (destructor)symArray_dealloc, /*tp_dealloc*/
    0,                          /*tp_print*/
    0,                          /*tp_getattr*/
    0,                          /*tp_setattr*/
    0,                          /*tp_compare*/
    0,                          /*tp_repr*/
    0,                          /*tp_as_number*/
    0,                          /*tp_as_sequence*/
    0,                          /*tp_as_mapping*/
    0,                          /*tp_hash */
    0,                          /*tp_call*/
    0,                          /*tp_str*/
    0,                          /*tp_getattro*/
    0,                          /*tp_setattro*/
    0,                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "symArray objects",         /* tp_doc */
    0,		                    /* tp_traverse */
    0,		                    /* tp_clear */
    0,		                    /* tp_richcompare */
    0,		                    /* tp_weaklistoffset */
    0,		                    /* tp_iter */
    0,		                    /* tp_iternext */
    symArray_methods,           /* tp_methods */
    symArray_members,           /* tp_members */
    symArray_getseters,         /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)symArray_init,    /* tp_init */
    0,                          /* tp_alloc */
    symArray_new,               /* tp_new */
};

/*  here follows the module initialization */

static struct PyMethodDef symmetricArrays_methods[] = {
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static char symmetricArrays_module_documentation[] = 
    "Module for tensor arrays with symmetric index pairs";

PyMODINIT_FUNC initsymmetricArrays(void) 
{
    PyObject* m;

    if (PyType_Ready(&symArrayType) < 0)
        return;

    m = Py_InitModule3("symmetricArrays", symmetricArrays_methods,
                        symmetricArrays_module_documentation);

    if (m == NULL)
      return;

    import_array();

    Py_INCREF(&symArrayType);
    PyModule_AddObject(m, "symArray", (PyObject *) &symArrayType);
}
