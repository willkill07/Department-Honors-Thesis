

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
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//custom matrix header
#include "Matrix.h"
#include "Functions.h"
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

/************************************************************/
// Functions


MatrixPair set_capture (const std::vector<cv::String> & im, unsigned dim);

Matrix test_capture (const std::string & test_path, unsigned dim);

Matrix orth_basis(const Matrix &A, const double cutoff);

double max_threshold(const Matrix &X);

void result (Matrix &X, const Matrix &Y, const double est);

/************************************************************/
int main(int argc, char *argv[]) 
{
    unsigned dim = 14400; //each image in our set is 120x120 
    const double cutoff = 0.8 * dim; //choose reduction cutoff <= 20%

    std::vector<cv::String> im; //vector of pathfiles to the set data elements
    cv::glob("./images/set*.png", im, false);
    std::string test_path = "./images/test/cheemse_test.png"; // path to the test image
    size_t set_size = im.size();

    //dealing with set data
    auto [Set, Mean] = set_capture(im, dim);
    Matrix A = Set - Mean;
    Matrix U = orth_basis(A, cutoff);
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

    for (int i = 0; i < set_size; ++i)
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

    for (int i = 0; i < set_size; ++i)
    {
        for (int j = 0; j < set_size; ++j)
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

    cv::Mat Grey;
    Matrix myGrey(dim, 1);

    for (size_t i = 0; i < set_size; ++i)
    {
        cvtColor(cv::imread(im[i]), Grey, cv::COLOR_BGR2GRAY); // reading in an image and turning it into grayscale
        std::copy(Grey.begin<uchar>(), Grey.end<uchar>(), myGrey.begin());
        //Make transition form cv::Mat to Matrix and make it an (m * n) x 1 column matrix
        Matrix::column_immerse(myGrey, Set, i);
        Mean = Mean + myGrey;
    }
    
    Mean *= 1 / set_size; // computing the mean face of each matrix


    Matrix Mean_Rect (dim, set_size); // Making a rectangular matrix of mean column vectors

    for (int i = 0; i < set_size; ++ i)
    {
        Matrix::column_immerse(Mean, Mean_Rect, i);
    }

    return MatrixPair(Set, Mean_Rect);
}

Matrix test_capture (const std::string & test_path, unsigned dim)
{
    cv::Mat Grey;
    Matrix myGrey(dim, 1);
    cvtColor(cv::imread(test_path), Grey, cv::COLOR_BGR2GRAY);
    std::copy(Grey.begin<uchar>(), Grey.end<uchar>(), myGrey.begin());
    //turn Grey into normal and make it an (m * n) x 1 column matrix
    return myGrey;
}

Matrix orth_basis(const Matrix &A, const double cutoff, unsigned dim)
{
    auto [U, S, VT] = singular_value_decomp(A); // Performing an SVD on A

    Matrix U_update (dim, cutoff);

    //fulling the orthogonal matrix with only the meaningful data columns
    for (int i = 0; i < cutoff; ++ i)
    {
        Matrix::column_immerse(U.column_extract(i), U_update, i);
    }

    return U.transpose();
}
