#pragma once
#include "AWIRecognition.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "WIR_OCR.h"

#define _DEBUG_MODE_WIR
#define _EXPEREMENTAL_MODE_WIR
#define BRIEF_DECTRIPTOR_SIZE 64

using namespace std;
using namespace cv;

class WIR01 :
	public AWIRecognition
{
private:
	WIRParam param;
	WIR_OCR ocr;
public:
	WIR01(void);
	WIR01(WIRParam param);
	virtual ~WIR01(void);
	//Recognize: number of potential candidates;
	virtual int Recognize(const char* file_path, WIRResult& result);
	virtual int Recognize(const char* file_path, vector<WIRResult>& results, unsigned int max_matches);
	//addTrainSamples; returs number of added samples
	virtual int addTrainSamples(vector<WIRTrainSample>& samples);
	// setRecognitionParam
	// DO NOT MODIFY WITHOUT BLOCKING. MULTITHREAD UNSAVE!!!!
	// all descriptors will be earased!!!
	virtual void setRecognitionParam(WIRParam param); 
	virtual int saveTrainingDB(const char* file_path);
	virtual int saveTrainingDBPartially(const vector<const char*>& directories, unsigned int filesPerDir, unsigned int descriptorsPerFile, const char* baseFileName);
	virtual int loadTrainingDB(const char* file_path);

	//Mobile Devices;
	int SaveBinary(const char* directory);
	int LoadBinary(const char* directory);
	//Set Callback function
	void setErrorCallback(WIRErrorCallback errorCallback);
	//clear
	void clear(void);
	//setting histogrammatic assistent
	void setHistogramUse(bool isOn) {param.useHistProcessing = isOn;};
	// STATIC METHODS
	//Generate update for other methods
	static bool GenerateUpdates(const WIRParam params, vector<WIRTrainSample>& samples, vector<const char*> directories, 
		unsigned int filesPerDir, unsigned int descriptorsPerFile, const char* baseFileName);
protected:
	virtual void train(void);
	int GetDescriptors();
	int ExtractDescriptors(const char* file_path, Mat& descriptors, vector<KeyPoint>& keypoints);
	int loadOCRParam() {return strlen(param.OCR_path)>0 ? ocr.loadTrainingDB(param.OCR_path):0;}; 
	int loadOCRParam(const char* path) {return ocr.loadTrainingDB(path);};
	bool loadedFromFile;
	void ImagePreProcessing( Mat& image);
	FeatureDetector* detector;
	DescriptorExtractor* extractor;
	FlannBasedMatcher* matcher;
	vector< vector<cv::KeyPoint> > dbKeyPoints;
	vector<Mat> dbDescriptors;
	vector<WIRTrainSample> trainSamples;
	WIRErrorCallback errorCallback;
	int maxClassLabel;
private:
	void WIRInternalPanic(int type = WIRE_GENERAL);
};

