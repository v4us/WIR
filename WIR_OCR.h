#pragma once
#include <opencv2/core/types_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include "WIRStruct.h"

using namespace cv;
using namespace std;

#define WIR_DO_NOT_DESTROY_KNEAREST
//#define _DEBUG_MODE_WIR_OCR

#define WIR_OCR_NOT_INITIALIZED -127;
#define WIR_OCR_MIN_BLOBS 2
class WIR_OCR
{
protected:
	static const int train_samples = 3;
	bool runAdditionalDilation;
	static const int classes = 10;
	static const int sizex = 20;
	static const int sizey = 30;
	static const int ImageSize = sizex * sizey;
	static const int minObjectHeight = 5;
	static const int minObjectWidth = 3;
	static const int sizeMorphElement1 = 33; //3%
	static const int sizeMorphElement2 = 100; //1%
	bool showDebugInformation;
	const double maxWidthF;// = 0.25;
	const double maxHeightF;// = 0.15;
	static const int darkBackgoundSeparater = 77; //0.3*255;
	int initialized;
	int labelExtraction;
	KNearest* knearest;
	WIRErrorCallback errorCallback;
	Mat trainData,trainClasses;

	void PreProcessImage(Mat *inImage,Mat *outImage,int sizex, int sizey);
	//int LearnFromImages(CvMat* trainData, CvMat* trainClasses);
	void RunSelfTest(KNearest& knn2,const char* pathToImages);
	void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs);
	int FindObjects(std::vector < std::vector<cv::Point2i > >& blobs, cv::Size imageSize, vector < cv::Rect >& contours );
	int LearnFromImages(Mat& trainData, Mat& trainClasses, const char* pathToImages );
	int InternalCurruptionCheck();
	void RegionColorize(cv::Size inputSize, cv::Mat& output,  std::vector < std::vector<cv::Point2i > >& blobs,
		int& maxX, int& minX, int& maxY, int& minY);
	int FindMaxBlob(const std::vector < std::vector<cv::Point2i > >& blobs) const;
	cv::Rect GetBlobRect(const std::vector<cv::Point2i >& blob) const;
public:
	void setLabelExtration(int labelExtraction) {this->labelExtraction = labelExtraction;};
	WIR_OCR(void);
	virtual ~WIR_OCR(void);
	//This function is only for internal testing this library.
	//DO NOT RUN in industrial use;
	int m(void);
	int saveTrainingDB(const char* file_path);
	int loadTrainingDB(const char* file_path);
	//Set Callback function
	void setErrorCallback(WIRErrorCallback errorCallback);
	//returning the list of all years
	int AnalyseImage(const Mat& image2, vector<unsigned int>& recognizedYears,cv::Rect* wineLabel = NULL);
	//returning the most probable year
	unsigned int AnalyseImage(const Mat& image,cv::Rect* wineLabel = NULL);
	int AnalyseImage(const char* image_path, vector<unsigned int>& recognizedYears, cv::Rect* wineLabel = NULL);
	inline void SetDebugInformationMode(bool showState){this->showDebugInformation = showState;};
	int isInit() {return initialized;};
private:
	void WIRInternalPanic(int type = WIRE_GENERAL);
};

struct RecognizedRegion
{
	unsigned int digit;
	cv::Rect rect;
	unsigned int group;

	RecognizedRegion(cv::Rect object, unsigned int digit, unsigned int group = 0)
	{
		this->digit = digit;
		this->rect = object;
		this->group = group;
	}
};