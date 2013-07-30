#pragma once
#include <stdio.h>
#include <iostream>
#include <string>
#include <set>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "WIR_OCR.h"
#include "AWIRecognition.h"
#include "WIR_clustering.h"

//#define _SAVE_CUTTED_IMAGIES
#define _DEBUG_MODE_WIR
#define _EXPEREMENTAL_MODE_WIR
//#define BRIEF_DECTRIPTOR_SIZE 64

#define LSH_FUNCTION_COUNT 6
//#define LSH_FUNCTION_COUNT 12 // recomended value
#define LSH_LENGTH 24
//#define LSH_LENGTH 20 //Default value

#define clusterCount 5
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
	//Param testing
	bool RecognitionTest(double& hitRate, double& firstHitRate, double& firstClassHitRate, double& classMatchHitRate);
	bool RecognitionTest(vector<WIRTrainSample>& trainSamples, double& hitRate,
		double& firstHitRate, double& firstClassHitRate, double& classMatchHitRate);
protected:
	virtual void train(void);
	int GetDescriptors();
	int ExtractDescriptors(const char* file_path, Mat& descriptors, vector<KeyPoint>& keypoints);
	int loadOCRParam() {return strlen(param.OCR_path)>0 ? ocr.loadTrainingDB(param.OCR_path):0;}; 
	int loadOCRParam(const char* path) {return ocr.loadTrainingDB(path);};
	bool loadedFromFile;
	bool useClustering;
	bool cropping;
	bool preCropping;
	bool afterClusteringCropping;
	bool pushSameClassImages;
	void ImagePreProcessing( Mat& image);
	FeatureDetector* detector;
	DescriptorExtractor* extractor;
	FlannBasedMatcher* matcher;
	BFMatcher* clusterMatcher;
	//FlannBasedMatcher* clusterMatcher;
	//vector< vector<cv::KeyPoint> > dbKeyPoints;
	vector<Mat> dbDescriptors;
	vector<Mat> clusteredDescriptors;
	vector<WIRTrainSample> trainSamples;
	WIRErrorCallback errorCallback;
	int maxClassLabel;
	const float rationalSeparater;
private:
	void WIRInternalPanic(int type = WIRE_GENERAL);
public:
	void SetUseClustering(bool value) {useClustering = value; if(value) ResetClusters();};
	void SetPreCropping(bool value) {preCropping = value;};
	void SetPushSameClassImages(bool value) {pushSameClassImages = value;};
	void SetAfterCCropping(bool value) {afterClusteringCropping = value;};
	void ResetClusters(void) {if (!useClustering) return; clusteredDescriptors.clear(); this->train();};
	void SetCropping(bool value){cropping = value;};
};

