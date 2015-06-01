/*****************************************************
 * CG Solver (HPC Software Lab)
 *
 * Parallel Programming Models for Applications in the 
 * Area of High-Performance Computation
 *====================================================
 * IT Center (ITC)
 * RWTH Aachen University, Germany
 * Author: Tim Cramer (cramer@itc.rwth-aachen.de)
 * Date: 2010 - 2015
 *****************************************************/


#ifndef __SOLVER_H__
#define __SOLVER_H__

#include "def.h"

#ifdef CUDA
 #include <cuda.h>
 #include <cuda_runtime.h>
#endif


#ifdef __cplusplus
	extern "C" {
#endif
	void vectorDot(const floatType* a, const floatType* b, const int n, floatType* ab);
	//__global__ void axpyCUDA(floatType a, floatType* x, int n, floatType* y);
	//__global__ void xpayCUDA(floatType* x, floatType a, int n, floatType* y);
	//__global__ void matvecCUDA(int n, int nnz, int maxNNZ, floatType* data, int* indices, int* length, floatType* x, floatType* y);
	void nrm2(const floatType* x, const int n, floatType* nrm);
	void cg(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* data_diag, const floatType* b, floatType* x, struct SolverConfig* sc);
#ifdef __cplusplus
	}
#endif


#endif
