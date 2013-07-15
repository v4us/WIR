#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#define _CRT_SECURE_NO_WARNINGS 1
#define _CRT_SECURE_NO_DEPRECATE 1
#pragma once
#include <stdio.h>
#include <iostream>
#include <string>
#include <stack>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "WIRStruct.h"

#define getBit(k,n) (k & (1<<(n)))
using namespace std;
using namespace cv;

//
class WIR_clustering 
{
public:
	static bool getCentroidsBRIEF(const cv::Mat& descriptors, cv::Mat& centroids, unsigned int countCentroids = 3);
private:
	virtual void make_me_abstract(void) = 0;
	static const int epsilon = 2;
	static const int maxIteration = 10000;
};

struct DistRecord
{
public:
	unsigned int i,j;
	int distance;
	DistRecord():i(0),j(0),distance(0){};
	DistRecord(unsigned int i, unsigned int j, int distance){this->i = i; this->j=j; this->distance = distance;};
};

