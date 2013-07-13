#pragma once
#include "WIRStruct.h"
#include <vector>

using namespace std;
class AWIRecognition
{
public:
	//Recognize: number of pottential candidates;
	virtual int Recognize(const char* file_path, WIRResult& result) = 0;
	virtual int Recognize(const char* file_path, vector<WIRResult>& results, unsigned int max_matches) = 0;
	//addTrainSamples; returs number of added samples
	virtual int addTrainSamples(vector<WIRTrainSample>& samples) = 0;
	virtual void setRecognitionParam(WIRParam param) = 0;
	virtual int saveTrainingDB(const char* file_path) = 0;
	virtual int loadTrainingDB(const char* file_path) = 0;
protected:
	virtual void train(void) = 0;
};

