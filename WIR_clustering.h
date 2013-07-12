#pragma once
#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "WIRStruct.h""


using namespace std;
using namespace cv;

class WIR_clustering 
{
private:
	WIRParam param;
public:
//
private:
	void make_me_abstract(void) = 0;
};

