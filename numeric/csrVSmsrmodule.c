/* File: csrVSmsrmodule.c
 * This file is auto-generated with f2py (version:2_5237).
 * f2py is a Fortran to Python Interface Generator (FPIG), Second Edition,
 * written by Pearu Peterson <pearu@cens.ioc.ee>.
 * See http://cens.ioc.ee/projects/f2py2e/
 * Generation date: Tue Oct 27 11:51:19 2009
 * $Revision:$
 * $Date:$
 * Do not edit this file directly unless you know what you are doing!!!
 */
#ifdef __cplusplus
extern "C" {
#endif

/*********************** See f2py2e/cfuncs.py: includes ***********************/
#include "Python.h"
#include "fortranobject.h"
#include <math.h>

/**************** See f2py2e/rules.py: mod_rules['modulebody'] ****************/
static PyObject *csrVSmsr_error;
static PyObject *csrVSmsr_module;

/*********************** See f2py2e/cfuncs.py: typedefs ***********************/
/*need_typedefs*/

/****************** See f2py2e/cfuncs.py: typedefs_generated ******************/
/*need_typedefs_generated*/

/********************** See f2py2e/cfuncs.py: cppmacros **********************/
#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F
#else
#define F_FUNC(f,F) _##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F##_
#else
#define F_FUNC(f,F) _##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)
#else
#define F_FUNC_US(f,F) F_FUNC(f,F)
#endif

#define rank(var) var ## _Rank
#define shape(var,dim) var ## _Dims[dim]
#define old_rank(var) (((PyArrayObject *)(capi_ ## var ## _tmp))->nd)
#define old_shape(var,dim) (((PyArrayObject *)(capi_ ## var ## _tmp))->dimensions[dim])
#define fshape(var,dim) shape(var,rank(var)-dim-1)
#define len(var) shape(var,0)
#define flen(var) fshape(var,0)
#define size(var) PyArray_SIZE((PyArrayObject *)(capi_ ## var ## _tmp))
/* #define index(i) capi_i ## i */
#define slen(var) capi_ ## var ## _len

#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,"debug-capi:"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \
  PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\
  fprintf(stderr,"\n");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif

#define max(a,b) ((a > b) ? (a) : (b))
#define min(a,b) ((a < b) ? (a) : (b))
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? (a) : (b))
#endif


/************************ See f2py2e/cfuncs.py: cfuncs ************************/
static int int_from_pyobj(int* v,PyObject *obj,const char *errmess) {
  PyObject* tmp = NULL;
  if (PyInt_Check(obj)) {
    *v = (int)PyInt_AS_LONG(obj);
    return 1;
  }
  tmp = PyNumber_Int(obj);
  if (tmp) {
    *v = PyInt_AS_LONG(tmp);
    Py_DECREF(tmp);
    return 1;
  }
  if (PyComplex_Check(obj))
    tmp = PyObject_GetAttrString(obj,"real");
  else if (PyString_Check(obj))
    /*pass*/;
  else if (PySequence_Check(obj))
    tmp = PySequence_GetItem(obj,0);
  if (tmp) {
    PyErr_Clear();
    if (int_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
    Py_DECREF(tmp);
  }
  {
    PyObject* err = PyErr_Occurred();
    if (err==NULL) err = csrVSmsr_error;
    PyErr_SetString(err,errmess);
  }
  return 0;
}


/********************* See f2py2e/cfuncs.py: userincludes *********************/
/*need_userincludes*/

/********************* See f2py2e/capi_rules.py: usercode *********************/


/* See f2py2e/rules.py */
extern void F_FUNC(csrmsr,CSRMSR)(int*,double*,int*,int*,double*,int*,double*,int*);
extern void F_FUNC(msrcsr,MSRCSR)(int*,double*,int*,double*,int*,int*,double*,int*);
extern void F_FUNC(csrcsc,CSRCSC)(int*,int*,int*,double*,int*,int*,double*,int*,int*);
/*eof externroutines*/

/******************** See f2py2e/capi_rules.py: usercode1 ********************/


/******************* See f2py2e/cb_rules.py: buildcallback *******************/
/*need_callbacks*/

/*********************** See f2py2e/rules.py: buildapi ***********************/

/*********************************** csrmsr ***********************************/
static char doc_f2py_rout_csrVSmsr_csrmsr[] = "\
Function signature:\n\
  ao,jao = csrmsr(a,ja,ia)\n\
Required arguments:\n"
"  a : input rank-1 array('d') with bounds (*)\n"
"  ja : input rank-1 array('i') with bounds (*)\n"
"  ia : input rank-1 array('i') with bounds (*)\n"
"Return objects:\n"
"  ao : rank-1 array('d') with bounds (len(a)+1)\n"
"  jao : rank-1 array('i') with bounds (len(a)+1)";
/* extern void F_FUNC(csrmsr,CSRMSR)(int*,double*,int*,int*,double*,int*,double*,int*); */
static PyObject *f2py_rout_csrVSmsr_csrmsr(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(int*,double*,int*,int*,double*,int*,double*,int*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  int n = 0;
  double *a = NULL;
  npy_intp a_Dims[1] = {-1};
  const int a_Rank = 1;
  PyArrayObject *capi_a_tmp = NULL;
  int capi_a_intent = 0;
  PyObject *a_capi = Py_None;
  int *ja = NULL;
  npy_intp ja_Dims[1] = {-1};
  const int ja_Rank = 1;
  PyArrayObject *capi_ja_tmp = NULL;
  int capi_ja_intent = 0;
  PyObject *ja_capi = Py_None;
  int *ia = NULL;
  npy_intp ia_Dims[1] = {-1};
  const int ia_Rank = 1;
  PyArrayObject *capi_ia_tmp = NULL;
  int capi_ia_intent = 0;
  PyObject *ia_capi = Py_None;
  double *ao = NULL;
  npy_intp ao_Dims[1] = {-1};
  const int ao_Rank = 1;
  PyArrayObject *capi_ao_tmp = NULL;
  int capi_ao_intent = 0;
  int *jao = NULL;
  npy_intp jao_Dims[1] = {-1};
  const int jao_Rank = 1;
  PyArrayObject *capi_jao_tmp = NULL;
  int capi_jao_intent = 0;
  double *wk = NULL;
  npy_intp wk_Dims[1] = {-1};
  const int wk_Rank = 1;
  PyArrayObject *capi_wk_tmp = NULL;
  int capi_wk_intent = 0;
  int *iwk = NULL;
  npy_intp iwk_Dims[1] = {-1};
  const int iwk_Rank = 1;
  PyArrayObject *capi_iwk_tmp = NULL;
  int capi_iwk_intent = 0;
  static char *capi_kwlist[] = {"a","ja","ia",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOO|:csrVSmsr.csrmsr",\
    capi_kwlist,&a_capi,&ja_capi,&ia_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable a */
  ;
  capi_a_intent |= F2PY_INTENT_IN;
  capi_a_tmp = array_from_pyobj(PyArray_DOUBLE,a_Dims,a_Rank,capi_a_intent,a_capi);
  if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting 1st argument `a' of csrVSmsr.csrmsr to C/Fortran array" );
  } else {
    a = (double *)(capi_a_tmp->data);

  /* Processing variable ia */
  ;
  capi_ia_intent |= F2PY_INTENT_IN;
  capi_ia_tmp = array_from_pyobj(PyArray_INT,ia_Dims,ia_Rank,capi_ia_intent,ia_capi);
  if (capi_ia_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting 3rd argument `ia' of csrVSmsr.csrmsr to C/Fortran array" );
  } else {
    ia = (int *)(capi_ia_tmp->data);

  /* Processing variable ja */
  ;
  capi_ja_intent |= F2PY_INTENT_IN;
  capi_ja_tmp = array_from_pyobj(PyArray_INT,ja_Dims,ja_Rank,capi_ja_intent,ja_capi);
  if (capi_ja_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting 2nd argument `ja' of csrVSmsr.csrmsr to C/Fortran array" );
  } else {
    ja = (int *)(capi_ja_tmp->data);

  /* Processing variable ao */
  ao_Dims[0]=len(a)+1;
  capi_ao_intent |= F2PY_INTENT_HIDE|F2PY_INTENT_OUT;
  capi_ao_tmp = array_from_pyobj(PyArray_DOUBLE,ao_Dims,ao_Rank,capi_ao_intent,Py_None);
  if (capi_ao_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting hidden `ao' of csrVSmsr.csrmsr to C/Fortran array" );
  } else {
    ao = (double *)(capi_ao_tmp->data);

  /* Processing variable n */
  n = len(ia)-1;
  /* Processing variable jao */
  jao_Dims[0]=len(a)+1;
  capi_jao_intent |= F2PY_INTENT_HIDE|F2PY_INTENT_OUT;
  capi_jao_tmp = array_from_pyobj(PyArray_INT,jao_Dims,jao_Rank,capi_jao_intent,Py_None);
  if (capi_jao_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting hidden `jao' of csrVSmsr.csrmsr to C/Fortran array" );
  } else {
    jao = (int *)(capi_jao_tmp->data);

  /* Processing variable wk */
  wk_Dims[0]=n;
  capi_wk_intent |= F2PY_INTENT_HIDE;
  capi_wk_tmp = array_from_pyobj(PyArray_DOUBLE,wk_Dims,wk_Rank,capi_wk_intent,Py_None);
  if (capi_wk_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting hidden `wk' of csrVSmsr.csrmsr to C/Fortran array" );
  } else {
    wk = (double *)(capi_wk_tmp->data);

  /* Processing variable iwk */
  iwk_Dims[0]=n + 1;
  capi_iwk_intent |= F2PY_INTENT_HIDE;
  capi_iwk_tmp = array_from_pyobj(PyArray_INT,iwk_Dims,iwk_Rank,capi_iwk_intent,Py_None);
  if (capi_iwk_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting hidden `iwk' of csrVSmsr.csrmsr to C/Fortran array" );
  } else {
    iwk = (int *)(capi_iwk_tmp->data);

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(&n,a,ja,ia,ao,jao,wk,iwk);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("NN",capi_ao_tmp,capi_jao_tmp);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
    Py_XDECREF(capi_iwk_tmp);
  }  /*if (capi_iwk_tmp == NULL) ... else of iwk*/
  /* End of cleaning variable iwk */
    Py_XDECREF(capi_wk_tmp);
  }  /*if (capi_wk_tmp == NULL) ... else of wk*/
  /* End of cleaning variable wk */
  }  /*if (capi_jao_tmp == NULL) ... else of jao*/
  /* End of cleaning variable jao */
  /* End of cleaning variable n */
  }  /*if (capi_ao_tmp == NULL) ... else of ao*/
  /* End of cleaning variable ao */
  if((PyObject *)capi_ja_tmp!=ja_capi) {
    Py_XDECREF(capi_ja_tmp); }
  }  /*if (capi_ja_tmp == NULL) ... else of ja*/
  /* End of cleaning variable ja */
  if((PyObject *)capi_ia_tmp!=ia_capi) {
    Py_XDECREF(capi_ia_tmp); }
  }  /*if (capi_ia_tmp == NULL) ... else of ia*/
  /* End of cleaning variable ia */
  if((PyObject *)capi_a_tmp!=a_capi) {
    Py_XDECREF(capi_a_tmp); }
  }  /*if (capi_a_tmp == NULL) ... else of a*/
  /* End of cleaning variable a */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/******************************* end of csrmsr *******************************/

/*********************************** msrcsr ***********************************/
static char doc_f2py_rout_csrVSmsr_msrcsr[] = "\
Function signature:\n\
  ao,jao,iao = msrcsr(n,a,ja)\n\
Required arguments:\n"
"  n : input int\n"
"  a : input rank-1 array('d') with bounds (*)\n"
"  ja : input rank-1 array('i') with bounds (*)\n"
"Return objects:\n"
"  ao : rank-1 array('d') with bounds (len(a))\n"
"  jao : rank-1 array('i') with bounds (len(a))\n"
"  iao : rank-1 array('i') with bounds (n + 1)";
/* extern void F_FUNC(msrcsr,MSRCSR)(int*,double*,int*,double*,int*,int*,double*,int*); */
static PyObject *f2py_rout_csrVSmsr_msrcsr(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(int*,double*,int*,double*,int*,int*,double*,int*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  int n = 0;
  PyObject *n_capi = Py_None;
  double *a = NULL;
  npy_intp a_Dims[1] = {-1};
  const int a_Rank = 1;
  PyArrayObject *capi_a_tmp = NULL;
  int capi_a_intent = 0;
  PyObject *a_capi = Py_None;
  int *ja = NULL;
  npy_intp ja_Dims[1] = {-1};
  const int ja_Rank = 1;
  PyArrayObject *capi_ja_tmp = NULL;
  int capi_ja_intent = 0;
  PyObject *ja_capi = Py_None;
  double *ao = NULL;
  npy_intp ao_Dims[1] = {-1};
  const int ao_Rank = 1;
  PyArrayObject *capi_ao_tmp = NULL;
  int capi_ao_intent = 0;
  int *jao = NULL;
  npy_intp jao_Dims[1] = {-1};
  const int jao_Rank = 1;
  PyArrayObject *capi_jao_tmp = NULL;
  int capi_jao_intent = 0;
  int *iao = NULL;
  npy_intp iao_Dims[1] = {-1};
  const int iao_Rank = 1;
  PyArrayObject *capi_iao_tmp = NULL;
  int capi_iao_intent = 0;
  double *wk = NULL;
  npy_intp wk_Dims[1] = {-1};
  const int wk_Rank = 1;
  PyArrayObject *capi_wk_tmp = NULL;
  int capi_wk_intent = 0;
  int *iwk = NULL;
  npy_intp iwk_Dims[1] = {-1};
  const int iwk_Rank = 1;
  PyArrayObject *capi_iwk_tmp = NULL;
  int capi_iwk_intent = 0;
  static char *capi_kwlist[] = {"n","a","ja",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOO|:csrVSmsr.msrcsr",\
    capi_kwlist,&n_capi,&a_capi,&ja_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable a */
  ;
  capi_a_intent |= F2PY_INTENT_IN;
  capi_a_tmp = array_from_pyobj(PyArray_DOUBLE,a_Dims,a_Rank,capi_a_intent,a_capi);
  if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting 2nd argument `a' of csrVSmsr.msrcsr to C/Fortran array" );
  } else {
    a = (double *)(capi_a_tmp->data);

  /* Processing variable n */
    f2py_success = int_from_pyobj(&n,n_capi,"csrVSmsr.msrcsr() 1st argument (n) can't be converted to int");
  if (f2py_success) {
  /* Processing variable ja */
  ;
  capi_ja_intent |= F2PY_INTENT_IN;
  capi_ja_tmp = array_from_pyobj(PyArray_INT,ja_Dims,ja_Rank,capi_ja_intent,ja_capi);
  if (capi_ja_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting 3rd argument `ja' of csrVSmsr.msrcsr to C/Fortran array" );
  } else {
    ja = (int *)(capi_ja_tmp->data);

  /* Processing variable iao */
  iao_Dims[0]=n + 1;
  capi_iao_intent |= F2PY_INTENT_HIDE|F2PY_INTENT_OUT;
  capi_iao_tmp = array_from_pyobj(PyArray_INT,iao_Dims,iao_Rank,capi_iao_intent,Py_None);
  if (capi_iao_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting hidden `iao' of csrVSmsr.msrcsr to C/Fortran array" );
  } else {
    iao = (int *)(capi_iao_tmp->data);

  /* Processing variable wk */
  wk_Dims[0]=n;
  capi_wk_intent |= F2PY_INTENT_HIDE;
  capi_wk_tmp = array_from_pyobj(PyArray_DOUBLE,wk_Dims,wk_Rank,capi_wk_intent,Py_None);
  if (capi_wk_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting hidden `wk' of csrVSmsr.msrcsr to C/Fortran array" );
  } else {
    wk = (double *)(capi_wk_tmp->data);

  /* Processing variable iwk */
  iwk_Dims[0]=n + 1;
  capi_iwk_intent |= F2PY_INTENT_HIDE;
  capi_iwk_tmp = array_from_pyobj(PyArray_INT,iwk_Dims,iwk_Rank,capi_iwk_intent,Py_None);
  if (capi_iwk_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting hidden `iwk' of csrVSmsr.msrcsr to C/Fortran array" );
  } else {
    iwk = (int *)(capi_iwk_tmp->data);

  /* Processing variable ao */
  ao_Dims[0]=len(a);
  capi_ao_intent |= F2PY_INTENT_HIDE|F2PY_INTENT_OUT;
  capi_ao_tmp = array_from_pyobj(PyArray_DOUBLE,ao_Dims,ao_Rank,capi_ao_intent,Py_None);
  if (capi_ao_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting hidden `ao' of csrVSmsr.msrcsr to C/Fortran array" );
  } else {
    ao = (double *)(capi_ao_tmp->data);

  /* Processing variable jao */
  jao_Dims[0]=len(a);
  capi_jao_intent |= F2PY_INTENT_HIDE|F2PY_INTENT_OUT;
  capi_jao_tmp = array_from_pyobj(PyArray_INT,jao_Dims,jao_Rank,capi_jao_intent,Py_None);
  if (capi_jao_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting hidden `jao' of csrVSmsr.msrcsr to C/Fortran array" );
  } else {
    jao = (int *)(capi_jao_tmp->data);

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(&n,a,ja,ao,jao,iao,wk,iwk);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("NNN",capi_ao_tmp,capi_jao_tmp,capi_iao_tmp);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  }  /*if (capi_jao_tmp == NULL) ... else of jao*/
  /* End of cleaning variable jao */
  }  /*if (capi_ao_tmp == NULL) ... else of ao*/
  /* End of cleaning variable ao */
    Py_XDECREF(capi_iwk_tmp);
  }  /*if (capi_iwk_tmp == NULL) ... else of iwk*/
  /* End of cleaning variable iwk */
    Py_XDECREF(capi_wk_tmp);
  }  /*if (capi_wk_tmp == NULL) ... else of wk*/
  /* End of cleaning variable wk */
  }  /*if (capi_iao_tmp == NULL) ... else of iao*/
  /* End of cleaning variable iao */
  if((PyObject *)capi_ja_tmp!=ja_capi) {
    Py_XDECREF(capi_ja_tmp); }
  }  /*if (capi_ja_tmp == NULL) ... else of ja*/
  /* End of cleaning variable ja */
  } /*if (f2py_success) of n*/
  /* End of cleaning variable n */
  if((PyObject *)capi_a_tmp!=a_capi) {
    Py_XDECREF(capi_a_tmp); }
  }  /*if (capi_a_tmp == NULL) ... else of a*/
  /* End of cleaning variable a */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/******************************* end of msrcsr *******************************/

/*********************************** csrcsc ***********************************/
static char doc_f2py_rout_csrVSmsr_csrcsc[] = "\
Function signature:\n\
  ao,jao,iao = csrcsc(a,ja,ia,[job,ipos])\n\
Required arguments:\n"
"  a : input rank-1 array('d') with bounds (*)\n"
"  ja : input rank-1 array('i') with bounds (*)\n"
"  ia : input rank-1 array('i') with bounds (*)\n"
"Optional arguments:\n"
"  job := 1 input int\n"
"  ipos := 1.0 input int\n"
"Return objects:\n"
"  ao : rank-1 array('d') with bounds (len(a))\n"
"  jao : rank-1 array('i') with bounds (len(a))\n"
"  iao : rank-1 array('i') with bounds (len(ia))";
/* extern void F_FUNC(csrcsc,CSRCSC)(int*,int*,int*,double*,int*,int*,double*,int*,int*); */
static PyObject *f2py_rout_csrVSmsr_csrcsc(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(int*,int*,int*,double*,int*,int*,double*,int*,int*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  int n = 0;
  int job = 0;
  PyObject *job_capi = Py_None;
  int ipos = 0;
  PyObject *ipos_capi = Py_None;
  double *a = NULL;
  npy_intp a_Dims[1] = {-1};
  const int a_Rank = 1;
  PyArrayObject *capi_a_tmp = NULL;
  int capi_a_intent = 0;
  PyObject *a_capi = Py_None;
  int *ja = NULL;
  npy_intp ja_Dims[1] = {-1};
  const int ja_Rank = 1;
  PyArrayObject *capi_ja_tmp = NULL;
  int capi_ja_intent = 0;
  PyObject *ja_capi = Py_None;
  int *ia = NULL;
  npy_intp ia_Dims[1] = {-1};
  const int ia_Rank = 1;
  PyArrayObject *capi_ia_tmp = NULL;
  int capi_ia_intent = 0;
  PyObject *ia_capi = Py_None;
  double *ao = NULL;
  npy_intp ao_Dims[1] = {-1};
  const int ao_Rank = 1;
  PyArrayObject *capi_ao_tmp = NULL;
  int capi_ao_intent = 0;
  int *jao = NULL;
  npy_intp jao_Dims[1] = {-1};
  const int jao_Rank = 1;
  PyArrayObject *capi_jao_tmp = NULL;
  int capi_jao_intent = 0;
  int *iao = NULL;
  npy_intp iao_Dims[1] = {-1};
  const int iao_Rank = 1;
  PyArrayObject *capi_iao_tmp = NULL;
  int capi_iao_intent = 0;
  static char *capi_kwlist[] = {"a","ja","ia","job","ipos",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOO|OO:csrVSmsr.csrcsc",\
    capi_kwlist,&a_capi,&ja_capi,&ia_capi,&job_capi,&ipos_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable a */
  ;
  capi_a_intent |= F2PY_INTENT_IN;
  capi_a_tmp = array_from_pyobj(PyArray_DOUBLE,a_Dims,a_Rank,capi_a_intent,a_capi);
  if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting 1st argument `a' of csrVSmsr.csrcsc to C/Fortran array" );
  } else {
    a = (double *)(capi_a_tmp->data);

  /* Processing variable ipos */
  if (ipos_capi == Py_None) ipos = 1.0; else
    f2py_success = int_from_pyobj(&ipos,ipos_capi,"csrVSmsr.csrcsc() 2nd keyword (ipos) can't be converted to int");
  if (f2py_success) {
  /* Processing variable job */
  if (job_capi == Py_None) job = 1; else
    f2py_success = int_from_pyobj(&job,job_capi,"csrVSmsr.csrcsc() 1st keyword (job) can't be converted to int");
  if (f2py_success) {
  /* Processing variable ia */
  ;
  capi_ia_intent |= F2PY_INTENT_IN;
  capi_ia_tmp = array_from_pyobj(PyArray_INT,ia_Dims,ia_Rank,capi_ia_intent,ia_capi);
  if (capi_ia_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting 3rd argument `ia' of csrVSmsr.csrcsc to C/Fortran array" );
  } else {
    ia = (int *)(capi_ia_tmp->data);

  /* Processing variable ja */
  ;
  capi_ja_intent |= F2PY_INTENT_IN;
  capi_ja_tmp = array_from_pyobj(PyArray_INT,ja_Dims,ja_Rank,capi_ja_intent,ja_capi);
  if (capi_ja_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting 2nd argument `ja' of csrVSmsr.csrcsc to C/Fortran array" );
  } else {
    ja = (int *)(capi_ja_tmp->data);

  /* Processing variable iao */
  iao_Dims[0]=len(ia);
  capi_iao_intent |= F2PY_INTENT_HIDE|F2PY_INTENT_OUT;
  capi_iao_tmp = array_from_pyobj(PyArray_INT,iao_Dims,iao_Rank,capi_iao_intent,Py_None);
  if (capi_iao_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting hidden `iao' of csrVSmsr.csrcsc to C/Fortran array" );
  } else {
    iao = (int *)(capi_iao_tmp->data);

  /* Processing variable ao */
  ao_Dims[0]=len(a);
  capi_ao_intent |= F2PY_INTENT_HIDE|F2PY_INTENT_OUT;
  capi_ao_tmp = array_from_pyobj(PyArray_DOUBLE,ao_Dims,ao_Rank,capi_ao_intent,Py_None);
  if (capi_ao_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting hidden `ao' of csrVSmsr.csrcsc to C/Fortran array" );
  } else {
    ao = (double *)(capi_ao_tmp->data);

  /* Processing variable n */
  n = (len(ia)-1);
  /* Processing variable jao */
  jao_Dims[0]=len(a);
  capi_jao_intent |= F2PY_INTENT_HIDE|F2PY_INTENT_OUT;
  capi_jao_tmp = array_from_pyobj(PyArray_INT,jao_Dims,jao_Rank,capi_jao_intent,Py_None);
  if (capi_jao_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(csrVSmsr_error,"failed in converting hidden `jao' of csrVSmsr.csrcsc to C/Fortran array" );
  } else {
    jao = (int *)(capi_jao_tmp->data);

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(&n,&job,&ipos,a,ja,ia,ao,jao,iao);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("NNN",capi_ao_tmp,capi_jao_tmp,capi_iao_tmp);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  }  /*if (capi_jao_tmp == NULL) ... else of jao*/
  /* End of cleaning variable jao */
  /* End of cleaning variable n */
  }  /*if (capi_ao_tmp == NULL) ... else of ao*/
  /* End of cleaning variable ao */
  }  /*if (capi_iao_tmp == NULL) ... else of iao*/
  /* End of cleaning variable iao */
  if((PyObject *)capi_ja_tmp!=ja_capi) {
    Py_XDECREF(capi_ja_tmp); }
  }  /*if (capi_ja_tmp == NULL) ... else of ja*/
  /* End of cleaning variable ja */
  if((PyObject *)capi_ia_tmp!=ia_capi) {
    Py_XDECREF(capi_ia_tmp); }
  }  /*if (capi_ia_tmp == NULL) ... else of ia*/
  /* End of cleaning variable ia */
  } /*if (f2py_success) of job*/
  /* End of cleaning variable job */
  } /*if (f2py_success) of ipos*/
  /* End of cleaning variable ipos */
  if((PyObject *)capi_a_tmp!=a_capi) {
    Py_XDECREF(capi_a_tmp); }
  }  /*if (capi_a_tmp == NULL) ... else of a*/
  /* End of cleaning variable a */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/******************************* end of csrcsc *******************************/
/*eof body*/

/******************* See f2py2e/f90mod_rules.py: buildhooks *******************/
/*need_f90modhooks*/

/************** See f2py2e/rules.py: module_rules['modulebody'] **************/

/******************* See f2py2e/common_rules.py: buildhooks *******************/

/*need_commonhooks*/

/**************************** See f2py2e/rules.py ****************************/

static FortranDataDef f2py_routine_defs[] = {
  {"csrmsr",-1,{{-1}},0,(char *)F_FUNC(csrmsr,CSRMSR),(f2py_init_func)f2py_rout_csrVSmsr_csrmsr,doc_f2py_rout_csrVSmsr_csrmsr},
  {"msrcsr",-1,{{-1}},0,(char *)F_FUNC(msrcsr,MSRCSR),(f2py_init_func)f2py_rout_csrVSmsr_msrcsr,doc_f2py_rout_csrVSmsr_msrcsr},
  {"csrcsc",-1,{{-1}},0,(char *)F_FUNC(csrcsc,CSRCSC),(f2py_init_func)f2py_rout_csrVSmsr_csrcsc,doc_f2py_rout_csrVSmsr_csrcsc},

/*eof routine_defs*/
  {NULL}
};

static PyMethodDef f2py_module_methods[] = {

  {NULL,NULL}
};

PyMODINIT_FUNC initcsrVSmsr(void) {
  int i;
  PyObject *m,*d, *s;
  m = csrVSmsr_module = Py_InitModule("csrVSmsr", f2py_module_methods);
  PyFortran_Type.ob_type = &PyType_Type;
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module csrVSmsr (failed to import numpy)"); return;}
  d = PyModule_GetDict(m);
  s = PyString_FromString("$Revision: $");
  PyDict_SetItemString(d, "__version__", s);
  s = PyString_FromString("This module 'csrVSmsr' is auto-generated with f2py (version:2_5237).\nFunctions:\n"
"  ao,jao = csrmsr(a,ja,ia)\n"
"  ao,jao,iao = msrcsr(n,a,ja)\n"
"  ao,jao,iao = csrcsc(a,ja,ia,job=1,ipos=1.0)\n"
".");
  PyDict_SetItemString(d, "__doc__", s);
  csrVSmsr_error = PyErr_NewException ("csrVSmsr.error", NULL, NULL);
  Py_DECREF(s);
  for(i=0;f2py_routine_defs[i].name!=NULL;i++)
    PyDict_SetItemString(d, f2py_routine_defs[i].name,PyFortranObject_NewAsAttr(&f2py_routine_defs[i]));



/*eof initf2pywraphooks*/
/*eof initf90modhooks*/

/*eof initcommonhooks*/


#ifdef F2PY_REPORT_ATEXIT
  if (! PyErr_Occurred())
    on_exit(f2py_report_on_exit,(void*)"csrVSmsr");
#endif

}
#ifdef __cplusplus
}
#endif
