#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#define _CRT_SECURE_NO_WARNINGS 1
#define _CRT_SECURE_NO_DEPRECATE 1

#ifndef _WIRSTRUCT_H_
#define _WIRSTRUCT_H_
#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <string>
#include <vector>
#include <sstream>

#ifndef _WIN32
#include "os_type.h"
#endif

struct WIRResult
{
public:
	char fileName[128]; //just image file's name
	char filePath[1024];//image path including file's name
	int assignedClassLabel;
	int classLabel;
	double propobility;
	double hist;
	unsigned int year;
	
	WIRResult& operator= (const WIRResult& right)
	{
		if (this == &right)
			return *this;
		
		fileName[0]=0;
		filePath[0]=0;
		strcpy(fileName,right.fileName);
		strcpy(filePath,right.filePath);
		assignedClassLabel = right.assignedClassLabel;
		classLabel = right.classLabel;
		propobility = right.propobility;
		year = right.year;
		hist = right.hist;

		return *this;
	}

	WIRResult(const WIRResult& obj)
	{
		fileName[0]=0;
		filePath[0]=0;
		strcpy(fileName,obj.fileName);
		strcpy(filePath,obj.filePath);
		assignedClassLabel = obj.assignedClassLabel;
		classLabel = obj.classLabel;
		propobility = obj.propobility;
		year = obj.year;
		hist = obj.hist;
	}
	WIRResult()
	{
		fileName[0] = 0;
		filePath[0] = 0;
		assignedClassLabel = 0;
		classLabel = 0;
		propobility = 0;
		year = 2000;
		hist = 1;
	};
	static std::ostringstream& output(std::ostringstream& stream, const WIRResult& res) 
	{
    stream <<"{"<<std::endl;
	stream << "\"fileName\" : \""<<res.fileName<<"\","<<std::endl;
	stream << "\"filePath\" : \""<<res.filePath<<"\","<<std::endl;
	stream << "\"assignedClassLabel\" : "<<res.assignedClassLabel<<" ,"<<std::endl;
	stream << "\"classLabel\" : "<<res.classLabel<<" ,"<<std::endl;
	stream << "\"propobility\" : "<<res.propobility<<" ,"<<std::endl;
	stream << "\"hist\" : "<<res.hist<<" ,"<<std::endl;
	stream << "\"year\" : "<<res.year<<std::endl;
	stream << "}"<<std::endl;
    return stream;
 };
	static std::ostringstream& vectorOutput(std::ostringstream& stream, const std::vector<WIRResult>& res)
{
	stream<<"["<<std::endl;
	if (res.size() != 0)
		for (size_t i =0; i<res.size(); i++)
		{
			WIRResult::output(stream,res[i]);
			if(i!=res.size()-1)
				stream <<" , " << std::endl;
		}
	stream<<"]"<<std::endl;
	return stream;
};
};




struct WIRTrainSample
{
public:
	char imagePath[1024]; //image path including file's name
	char imageName[128]; //just image file's name
	int classLabel;

	WIRTrainSample& operator= (const WIRTrainSample& right)
	{
		if (this == &right)
			return *this;

		imagePath[0]=0;
		imageName[0]=0;
		strcpy(imagePath,right.imagePath);
		strcpy(imageName,right.imagePath);
		classLabel = right.classLabel;
		return *this;
	}

	WIRTrainSample (const WIRTrainSample& obj)
	{
		imagePath[0]=0;
		imageName[0]=0;
		strcpy(imagePath,obj.imagePath);
		strcpy(imageName,obj.imageName);
		classLabel = obj.classLabel;
	}
	WIRTrainSample()
	{
		imagePath[0] = 0;
		imageName[0] = 0;
		classLabel = 0;
	}
	void write(cv::FileStorage& fs) const                        //Write serialization for this class
	{
		std::string tmpString(imagePath);
		std::string tmp2String(imageName);
		fs << "{" << "imagePath" << tmpString <<"imageName" <<tmp2String <<"classLabel" << classLabel << "}";
	}

	void read(const cv::FileNode& node)                          //Read serialization for this class
	{
		std::string tmpImagePath = (std::string)node["imagePath"];
		std::string tmpImageName = (std::string)node["imageName"];
		imagePath[0] = 0;
		imageName[0] = 0;
		strcpy(imagePath,tmpImagePath.c_str());
		strcpy(imageName,tmpImageName.c_str());
		classLabel = (int)node["classLabel"];
	}
};


enum WIRParamPreSetted {standartWIR = 0, fastWIR = 1, doubleFastWIR =2};
#define WIR_EL_NONE 0
#define WIR_EL_SOFT 1
#define WIR_EL_STRICT 2
struct WIRParam
{
public:
	double threshold;//minHessian;
	int bins;
	int useClassLabel;
	double goodSelectionMultilier;
	int useHistProcessing;
	char OCR_path[1000];
	char descriptorExtractorType[50];
	char detectorType[50];
	int labelExtraction; //noneWIR = 0, softWIR = 1, strictWIR = 2
	//Добавить параметр регулирую для проверки и оптимизации объединить

	void write(cv::FileStorage& fs) const                        //Write serialization for this class
	{
		std::string tmpOCRString(OCR_path);
		fs << "{" << "THR" << threshold << "bins" << bins 
			<< "useClassLabel" << useClassLabel << "GSM" << goodSelectionMultilier<<
			"OCR_path"<<tmpOCRString<<"useHistProcessing"<<useHistProcessing<<
			"detectorType"<<detectorType<<"descriptorExtractorType"<<descriptorExtractorType<<
			"labelExtraction"<<labelExtraction<<"}";
	}

	void read(const cv::FileNode& node)                          //Read serialization for this class
	{
		threshold = (double)node["THR"];
		bins = (int)node["bins"];
		useClassLabel = (int)node["useClassLabel"];
		goodSelectionMultilier = (double)node["GSM"];
		std::string tmpStr = (std::string)node["OCR_path"];
		OCR_path[0] = 0;
		std::strcpy(OCR_path,tmpStr.c_str());

		tmpStr = (std::string)node["descriptorExtractorType"];
		descriptorExtractorType[0] = 0;
		std::strcpy(descriptorExtractorType,tmpStr.c_str());

		tmpStr = (std::string)node["detectorType"];
		detectorType[0] = 0;
		std::strcpy(detectorType,tmpStr.c_str());

		useHistProcessing = (int)node["useHistProcessing"];
		labelExtraction =  ((int)node["labelExtraction"]);
	}

	WIRParam& operator= (const WIRParam& right)
	{
		if (this == &right)
			return *this;
		
		threshold = right.threshold;
		bins = right.bins;
		useClassLabel = right.useClassLabel;
		goodSelectionMultilier = right.goodSelectionMultilier;
		OCR_path[0] = 0;
		std::strcpy(OCR_path,right.OCR_path);
		detectorType[0] = 0;
		std::strcpy(detectorType,right.detectorType);
		descriptorExtractorType[0] = 0;
		std::strcpy(descriptorExtractorType,right.descriptorExtractorType);

		useHistProcessing = right.useHistProcessing;
		labelExtraction = right.labelExtraction;

		return *this;
	}

	WIRParam(const WIRParam& obj)
	{
		threshold = obj.threshold;
		bins = obj.bins;
		useClassLabel = obj.useClassLabel;
		goodSelectionMultilier = obj.goodSelectionMultilier;
		OCR_path[0] = 0;
		std::strcpy(OCR_path,obj.OCR_path);
		useHistProcessing = obj.useHistProcessing;
		detectorType[0] = 0;
		std::strcpy(detectorType,obj.detectorType);
		descriptorExtractorType[0] = 0;
		std::strcpy(descriptorExtractorType,obj.descriptorExtractorType);
		labelExtraction = obj.labelExtraction;
	}
	WIRParam(WIRParamPreSetted in)
	{
		switch (in)
		{
		case standartWIR:
			threshold = 400; // minHEssian
			useClassLabel = 0;
			bins = 2;
			useHistProcessing = 0;
			goodSelectionMultilier =3;
			OCR_path[0]=0;
			strcpy(OCR_path,"./OCR.XML");
			detectorType[0] = 0;
			strcpy(detectorType,"SURF");
			descriptorExtractorType[0] = 0;
			strcpy(descriptorExtractorType,"SURF");
			labelExtraction = 1;
			break;
		case fastWIR:
			threshold = 400; // minHEssian
			useClassLabel = 0;
			bins = 2;
			useHistProcessing = 0;
			goodSelectionMultilier =3;
			OCR_path[0]=0;
			strcpy(OCR_path,"./OCR.XML");
			detectorType[0] = 0;
			strcpy(detectorType,"SURF");
			labelExtraction = 1;
			descriptorExtractorType[0] = 0;
			strcpy(descriptorExtractorType,"BRIEF");
			break;
		case doubleFastWIR:
			threshold = 10; // minHEssian
			useClassLabel = 0;
			bins = 2;
			useHistProcessing = 0;
			goodSelectionMultilier =3;
			OCR_path[0]=0;
			strcpy(OCR_path,"./OCR.XML");
			detectorType[0] = 0;
			strcpy(detectorType,"FAST");
			descriptorExtractorType[0] = 0;
			labelExtraction = 1;
			strcpy(descriptorExtractorType,"BRIEF");
			break;
		}
	}
	WIRParam()
	{
		threshold = 400; // minHEssian
		useClassLabel = 0;
		bins = 2;
		useHistProcessing = 0;
		goodSelectionMultilier =3;
		OCR_path[0]=0;
		strcpy(OCR_path,"./OCR.XML");
		detectorType[0] = 0;
		strcpy(detectorType,"SURF");
		descriptorExtractorType[0] = 0;
		strcpy(descriptorExtractorType,"BRIEF");
		labelExtraction = 0;
	}

};
inline bool operator==(const WIRParam& lhs, const WIRParam& rhs)
{ 
	int  strComparerA = 0;
	strComparerA += abs(strcmp(lhs.OCR_path, rhs.OCR_path));
	strComparerA += abs(strcmp(lhs.descriptorExtractorType, rhs.descriptorExtractorType));
	strComparerA += abs(strcmp(lhs.detectorType, rhs.detectorType));
	if(strComparerA !=0)
		return false;
	if(lhs.threshold != rhs.threshold)
		return false;
	if(lhs.bins != rhs.bins)
		return false;
	if(lhs.useClassLabel != rhs.useClassLabel)
		return false;
	if(lhs.goodSelectionMultilier != rhs.goodSelectionMultilier)
		return false;
	if(lhs.useHistProcessing != rhs.useHistProcessing)
		return false;
	if (lhs.labelExtraction != rhs.labelExtraction)
		return false;
	return true;
}

typedef void (*WIRErrorCallback)(int);


//A negative number means a CRITICAL ERROR 
#define WIRE_GENERAL -1
#define WIRE_NOT_ENOUGH_MEMORY -2
#define WIRE_CANNOT_LOAD_IMAGE 1
#define WIRE_CANNOT_PROCESS_IO 2

//OpenCV auxiliary I/O function definition
static void write(cv::FileStorage& fs, const std::string&, const WIRParam& x)
{
	x.write(fs);
}

static void read(const cv::FileNode& node, WIRParam& x, const WIRParam& default_value = WIRParam())
{
	if(node.empty())
		x = default_value;
	else
		x.read(node);
}
static void write(cv::FileStorage& fs, const std::string&, const WIRTrainSample& x)
{
	x.write(fs);
}

static void read(const cv::FileNode& node, WIRTrainSample& x, const WIRTrainSample& default_value = WIRTrainSample())
{
	if(node.empty())
		x = default_value;
	else
		x.read(node);
}
#endif