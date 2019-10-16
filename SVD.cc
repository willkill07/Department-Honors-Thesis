/*
  Author     : Maksim Y. Melnichenko
  Title      : SVD Inplementation

  Project history and additional details are avaliable upon request.
/************************************************************/
// System includes

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

/************************************************************/
// Using declarations

using std::cin;
using std::cout;
using std::endl;
using std::pow;
using std::vector;
using std::sqrt;
using std::pow;

/************************************************************/
// Function prototypes/global vars/typedefs

void build(Matrix &A);

void tridiagonalizer(Matrix &B);

void similarityProducer(Matrix &WTW, double RSQ);

template <bool T1, bool O1, bool T2, bool O2>
void block_diagonal(MatrixT<T1, O1> &B, MatrixT<T2, O2> &Beta);

void divide(Matrix &B);

void qr(Matrix &B);

void qr_acceleration(Matrix &B, double e);

void printMatrix(Matrix A);

/************************************************************/
// Main function

int main(int argc, char *argv[]) {

  // int r = 2;
  // int c = 4;

  // if ( r < c)
  //{
  //    int i = r;
  //    r = c;
  //    c = i;
  //}
  // Since (A^T) * A == A * (A^T)

  // Matrix A (r, c);

  // build(A);
  // Initial Matrix
  // printMatrix(A);

  // auto B = A * A.transpose();

  // Building a numerical example

  Matrix TEST(4, 4);

  TEST(0, 0) = 4;
  TEST(0, 1) = 1;
  TEST(0, 2) = -2;
  TEST(0, 3) = 2;

  TEST(1, 0) = 1;
  TEST(1, 1) = 2;
  TEST(1, 2) = 0;
  TEST(1, 3) = 1;

  TEST(2, 0) = -2;
  TEST(2, 1) = 0;
  TEST(2, 2) = 3;
  TEST(2, 3) = -2;

  TEST(3, 0) = 2;
  TEST(3, 1) = 1;
  TEST(3, 2) = -2;
  TEST(3, 3) = -1;

  // A * (A ^ T)
  tridiagonalizer(TEST);
}


/*
Build - by reference, modifies the original matrix. Assignes a random
floating point number to each of its cells.
*/
void build(Matrix &A) {
  
  static std::minstd_rand rng{9999};
  static std::uniform_real_distribution<double> dist(-10.0, 10.0);
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < A.colms(); ++j) {
      A(i, j) = dist (rng);
    }
  }
}


/*
Tridiagonalizer - Implementation of Householder's method of turning a
symmetric matrix into a symmetric tridiagonal one. Modifies the original by refernce,
uses a helper routine for the similarity transformation matrix production.
*/
void tridiagonalizer(Matrix &B) {
  // producing a column vector, replicating the last column of B

  for (int k = 0; k < B.rows() - 2; ++k) {

    double alpha, RSQ = 0;
    Matrix W(B.rows(), k + 1);

    // for k = 0 ...  < n-2
    for (int i = k + 1; i < B.rows(); ++i) {
      alpha += std::pow(B(i, k), 2);
      W(i, k) = (0.5) * B(i, k);
    }

    alpha = -(std::sqrt(alpha));
    RSQ = (std::pow(alpha, 2) - (alpha * B(k + 1, k)));

    W(k, k) = 0;
    W(k + 1, k) = (0.5) * (B(k + 1, k) - alpha);

    auto WTW = W * W.transpose();
    similarityProducer(WTW, RSQ);

    B = WTW * B * WTW;
    printMatrix(B);
  }
}



/*
similarityProducer - by reference, takes in the matrix, which will further be used as a 
basis for the similarity transformation matrix.
*/
void similarityProducer(Matrix &WTW, double RSQ) {
  for (int i = 0; i < WTW.rows(); ++i) {
    for (int j = 0; j < WTW.colms(); ++j) {
      WTW(i, j) = -(4 / RSQ) * WTW(i, j);
    }
    WTW(i, i) += 1;
  }
}




/*
Divide - initial step of Cuppen's Divide and Conquer Eigenvalue Extraction algorithm.
/////////////////////////////////UNDER CONSTRUCTION/////////////////////////////////
*/
void divide(Matrix &B) {
  Matrix Beta(B.rows(), B.colms());
  block_diagonal(B, Beta);

  if (B.rows() < 2) {
    // return
  } else {
    Matrix hi = Matrix::cut(B, B.rows() / 2, 1);
    Matrix lo = Matrix::cut(B, B.colms() - (B.colms() / 2), 0);
    // return Beta + [(divide (hi))    (divide (lo))]
  }
}


/*
block_diagonal - routine that makes the original matrix, taken in by reference, 
block diagonal and additionally updates the "factored-out" matrix beta with corresponding
elements.
*/
template <bool T1, bool O1, bool T2, bool O2>
void block_diagonal(MatrixT<T1, O1> &B, MatrixT<T2, O2> &Beta) {

  double m = B.rows() / 2;
  double beta_value = B(m, m - 1);

  Beta(m, m) = Beta(m - 1, m - 1) = Beta(m, m - 1) = Beta(m - 1, m) =
      beta_value;

  B(m, m - 1) = B(m - 1, m) = 0;
  B(m, m) -= beta_value;
  B(m - 1, m - 1) -= beta_value;
}



/*
QR - The main portion of the QR Eigenvalue extraction algorithm. By reference, takes in the original
matrix and treats it as the matrix R which, at the end of the process, would be Upper Triangular. 
Finds the final states of R and orthogonal matrix Q, using rotation matrices, defined here as P.
Overall, performs only one iteration of the algorithm.
*/
void qr(Matrix &R)
{
  //initialize values of i, j, assuming B(i,i) above B(j,i)
  
  // Rotation Matrix
  auto Q = Matrix::identity(R.rows());  

  
  for(int i = 0, j = 1; j < R.rows(); ++i, ++j)
  {

    auto P = Matrix::identity(R.rows());

    P(i,i) = P(j,j) = R(i,i) / (sqrt( pow(R(i,i), 2) + pow(R(i, j), 2)));
    P(i, j) = R(i,j) / (sqrt( pow(R(i,i), 2) + pow(R(i, j), 2)));
    P(j, i) = - P(i, j);

    R = P * R;
    Q = P * Q;
    
  }

  //Final Reduction. The goal is to get the subdiagonal =~ 0
  R = Q * R;
}



/*
QR_acceleration - calles the QR routine to compress the subdiagonal elements to within the
range of e-range away from zero.
*/
void qr_acceleration(Matrix &M, double e)
{
  /*Perform several tests, determine if any of the subdiagonal values decreases faster.
  Based on that data, set up the comparison-call structure.
  */
}




/*
Printing - Simple routine, created for the testing purposes.
*/
void printMatrix(const Matrix A) {
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < A.colms(); ++j) {
      printf("%13.5f", A(i, j));
    }
    puts("");
  }
  puts("");
}
