

/************************************************************/
// System includes

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <math.h>
#include <random>
#include <stdexcept>
#include <stdio.h>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//custom matrix header
#include "Matrix.h"
#include "Functions.h"
#include "Timer.hpp" 
/************************************************************/
// Using declarations

using std::cin;
using std::cout;
using std::endl;
using std::pow;
using std::vector;
using std::sqrt;
using std::pow;

using namespace cv;
using Correction = std::pair<double, Matrix>;
using MatrixPair = std::pair<Matrix, Matrix>;
using MatrixTuple = std::tuple<Matrix, Matrix, Matrix>;
using MatrixQueue = std::queue<Matrix>; 

/************************************************************/
// Functions

void learner (const std::vector<cv::String> & im, const std::string & test_path, unsigned dim, const int cutoff);

MatrixPair set_capture (const std::vector<cv::String> & im, unsigned dim);

Matrix test_capture (const std::string & test_path, unsigned dim);

Matrix orth_basis(const Matrix &A, const int cutoff, unsigned dim);

double max_threshold(Matrix &X);

void result (Matrix &X, const Matrix &Y, const double est);

void playground (const std::string & file_path, const double cutoff);

/************************************************************/
int main() 
{
    //unsigned dim = 14400; //each image in our set is 120x120 
    const double cutoff = 1; // - in percentage - choose reduction cutoff <= 20%
    
    std::vector<cv::String> im; //vector of pathfiles to the set data elements
    cv::glob("./images/set/*.png", im, false);
    std::string test_path = "./images/test/120.png"; // path to the test image

    playground(test_path, cutoff);

    
}

void playground (const std::string & file_path, const double cutoff)
{
    cv::Mat Gray;
    Matrix myGray(120, 120);
    cvtColor(cv::imread(file_path), Gray, cv::COLOR_BGR2GRAY);
    std::transform(Gray.begin<uchar>(), Gray.end<uchar>(), myGray.begin(), [] (uchar val) { return val / 25.5; });

    MatrixQueue origQ = Matrix::partition(myGray, 4);
    MatrixQueue resQ;

    while (!origQ.empty())
    {   
        auto A = origQ.front();
        auto New = compression(A, cutoff);
        origQ.pop();
        resQ.push(New);
    }
    
    auto Processed = Matrix::assembling(resQ, myGray.rows(), myGray.colms());
    Processed *= 25.50;
    
    //std::copy(Processed.begin(), Processed.end(), Gray.begin<uchar>());
    cv::Mat Res(120, 120, CV_64F, Processed.begin());
    //cout << Res << "\n";
    imwrite( "./images/new_cheems.jpg", Res );
    
}


/*****************************************************/
/****************UNDER CONSTRUCTION*******************/
/*****************************************************/

void learner (const std::vector<cv::String> & im, const std::string & test_path, unsigned dim, const int cutoff)
{
    auto [Set, Mean] = set_capture(im, dim);
    auto A = Set - Mean;

    Matrix U = orth_basis(A, cutoff, dim); // I SEGFAULT HERE
    Matrix X = U * A; // The scalar projection of face - face_mean onto the base­faces - each column represents Xi
    const double est = max_threshold(X); 

    //dealing with test data
    Matrix T = test_capture(test_path, dim);
    Matrix B = T - Mean.column_extract(0);
    Matrix Y = U * B;

    result(X, Y, est);
}


void result (Matrix &X, const Matrix &Y, const double est)
{
    unsigned set_size = X.colms();

    for (unsigned i = 0; i < set_size; ++i)
    {
        auto buf = Matrix::magnitude(Y - X.column_extract(i));
        if (buf < est)
        {
            cout << "This image is in the dtatset\n";
            break;
        }
    }
    cout << "Unknown image\n";
}


double max_threshold(Matrix &X)
{
    unsigned set_size = X.colms();
    double est = 0;

    for (unsigned i = 0; i < set_size; ++i)
    {
        for (unsigned j = 0; j < set_size; ++j)
        {
            if (i != j)
            {
                auto buf = Matrix::magnitude(X.column_extract(j) - X.column_extract(i));
                if (buf > est)
                {
                    est = buf;
                }
            }
        }
    }
    return est;
}

MatrixPair set_capture (const std::vector<cv::String> & im, unsigned dim)
{
    size_t set_size = im.size();
    
    Matrix Mean (dim, 1); 
    Matrix Set (dim, set_size);

    cv::Mat Gray;
    Matrix myGray(dim, 1);

    for (unsigned i = 0; i < set_size; ++i)
    {
        cvtColor(cv::imread(im[i]), Gray, cv::COLOR_BGR2GRAY); // reading in an image and turning it into grayscale
        // Convert image to binary
        //Make transition form cv::Mat to Matrix and make it an (m * n) x 1 column matrix
        std::transform(Gray.begin<uchar>(), Gray.end<uchar>(), myGray.begin(), [] (uchar val) { return val / 255.0; });
        Set = Matrix::column_immerse(myGray, Set, i);
        Mean = Mean + myGray;
    }
    
    Mean *= 1 / (double) set_size; // computing the mean face of each matrix


    Matrix Mean_Rect (dim, set_size); // Making a rectangular matrix of mean column vectors

    for (unsigned i = 0; i < set_size; ++ i)
    {
        Mean_Rect = Matrix::column_immerse(Mean, Mean_Rect, i);
    }

    return MatrixPair(Set, Mean_Rect);
}

Matrix test_capture (const std::string & test_path, unsigned dim)
{
    cv::Mat Gray;
    Matrix myGray(dim, 1);
    cvtColor(cv::imread(test_path), Gray, cv::COLOR_BGR2GRAY);
    // Convert image to binary
    cout << Gray << "\n";
    //turn Grey into normal and make it an (m * n) x 1 column matrix
    std::transform(Gray.begin<uchar>(), Gray.end<uchar>(), myGray.begin(), [] (uchar val) { return val / 255.0; });
    return myGray;
}

Matrix orth_basis(const Matrix &A, const int cutoff, unsigned dim)
{
    auto [U, S, VT] = singular_value_decomp(A); // Performing an SVD on A
    Matrix U_update (dim, cutoff);

    //fulling the orthogonal matrix with only the meaningful data columns
    for (int i = 0; i < cutoff; ++ i)
    {
        U_update = Matrix::column_immerse(U.column_extract(i), U_update, i);
    }

    return U.transpose();
}