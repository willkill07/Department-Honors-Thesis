#ifndef FUNCTIONS_H_  
#define FUNCTIONS_H_  

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <random>
#include <stdexcept>
#include <stdio.h>
#include <utility>
#include <vector>

//custom matrix header
#include "Matrix.h"

using std::cin;
using std::cout;
using std::endl;
using std::pow;
using std::vector;
using std::sqrt;
using std::pow;

using Correction = std::pair<double, Matrix>;
using MatrixPair = std::pair<Matrix, Matrix>;
using MatrixTuple = std::tuple<Matrix, Matrix, Matrix>;

/************************************************************/
// Major function prototypes

void tridiagonalization(Matrix &Sym); //

MatrixPair eigen_decomp(Matrix &Sym); //

Matrix secular_solver(Matrix &Diag, const Correction &Cor);

Matrix evector_extract(const Matrix &Eig, const Matrix &Diag);

MatrixTuple singular_value_decomp(const Matrix &Init);

/************************************************************/
// Helper function prototypes

void populate_matrix(Matrix &Init); //

Correction block_diagonal(Matrix &Sym); //

Matrix initial_e_approx(const Matrix &Diag, const Correction &Cor);

Matrix s_construction(const Matrix &Init, const Matrix &Eig); //

Matrix u_construction(const Matrix &Init,  Matrix &Orth, const Matrix &S); //

void gram_schmidt(Matrix &U, int i); //

template <bool Asc, bool Row, bool T1, bool O1, bool T2, bool O2>
MatrixPair sorts(const MatrixT<T1, O1> &Diag, MatrixT<T2, O2> &Orth);

template <bool O, bool T>
void print_matrix(const MatrixT<O,T> &A); //

/************************************************************/
// Small mathematical functions

double secular_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y);

double secular_function_prime(const Matrix& Diag, const Matrix& Z, const double rho, const double y);

double psi(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k);

double phi(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k);

double psi_prime(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k);

double phi_prime(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k);

double g_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k);

double h_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k);
/************************************************************/

#endif