#include "stdafx.h"
#include "WIR01.h"


WIR01::WIR01(void):ocr()
{
	WIRParam tmpParam;
	tmpParam.threshold = 400; // minHEssian
	tmpParam.useClassLabel = 0;
	tmpParam.goodSelectionMultilier =3;
	tmpParam.useHistProcessing = 0;
	detector = NULL;
	extractor = NULL;
	matcher = NULL;
	tmpParam.OCR_path[0]=0;
	strcpy(tmpParam.OCR_path,"./OCR.XML");
	setRecognitionParam(tmpParam);
	errorCallback = NULL;
	maxClassLabel = 0;
	loadedFromFile = false;
#ifdef _DEBUG_MODE_WIR
	if(!loadOCRParam())
		cout<<"Cannot initialize OCR"<<endl;
#endif
#ifndef _DEBUG_MODE_WIR
	loadOCRParam();
#endif
}

WIR01::WIR01(WIRParam param):ocr()
{
	detector = NULL;
	extractor = NULL;
	matcher = NULL;
	setRecognitionParam(param);
	errorCallback = NULL;
	maxClassLabel = 0;
	loadedFromFile = false;
#ifdef _DEBUG_MODE_WIR
	if(!loadOCRParam())
		cout<<"Cannot initialize OCR"<<endl;
#endif
#ifndef _DEBUG_MODE_WIR
	loadOCRParam();
#endif
}

WIR01::~WIR01(void)
{
	if (!detector)
	{
		#ifdef _DEBUG_MODE_WIR
		std::cout<<"delete DETECTOR"<<endl;
		#endif
		delete detector;
	}
	if(!extractor)
	{
		#ifdef _DEBUG_MODE_WIR
		std::cout<<"delete EXTRACTOR"<<endl;
		#endif
		delete extractor;
	}
	if (!matcher)
	{
		#ifdef _DEBUG_MODE_WIR
		std::cout<<"delete MATCHER"<<endl;
		#endif
		matcher->clear();
		delete matcher;
	}
}

void WIR01::WIRInternalPanic(int type)
{
	#ifdef _DEBUG_MODE_WIR
		std::cout<<"Something has gone wrong!!! ERROR CODE:"<<type<<endl;
	#endif
	if(type<0)
	{
		dbDescriptors.clear();
		trainSamples.clear();
		loadedFromFile = false;
		delete detector;
		delete extractor;
		delete matcher;
	}
	if (errorCallback != NULL)
		errorCallback(type);
}

void WIR01::setErrorCallback(WIRErrorCallback errorCallback)
{
	this->errorCallback = errorCallback;
}

	//Recognize: number of pottential candidates;
int WIR01::Recognize(const char* file_path, WIRResult& result)
{
	vector<WIRResult> tmpResults;
	if(Recognize(file_path, tmpResults,1)>0)
	{
		result = tmpResults[0];
		return 1;
	}
	return 0;
};

int WIR01::Recognize(const char* file_path, vector<WIRResult>& results, unsigned int max_matches)
{
	if(file_path == NULL)
		{ 
		#ifdef _DEBUG_MODE_WIR
			std::cout<< " --(NOT DEFFINED) Error reading images " << std::endl;
			errorCallback(WIRE_CANNOT_LOAD_IMAGE);
		#endif		
		return -1; 
	};
	if(file_path[0] == 0)
		{ 
		#ifdef _DEBUG_MODE_WIR
			std::cout<< " --("<<file_path<<") Error reading images " << std::endl;
			errorCallback(WIRE_CANNOT_LOAD_IMAGE);
		#endif		
		return -1; 
	};
	Mat img = imread(file_path, IMREAD_GRAYSCALE ); //!!! IMREAD_GRAYSCALE
	if(!img.data )
	{ 
		#ifdef _DEBUG_MODE_WIR
			std::cout<< " --("<<file_path<<") Error reading images " << std::endl;
			errorCallback(WIRE_CANNOT_LOAD_IMAGE);
		#endif		
		return -1; 
	};
	unsigned int detectedYear = 999;
	cv::Rect labelArea;
	if( ocr.isInit())
	{
		detectedYear = ocr.AnalyseImage(img,&labelArea);
		img = img(labelArea);
		//Mat tmpImg = img(labelArea);
		imshow(",,,,",img);
		//waitKey(0);
	}
	ImagePreProcessing (img);
	vector<KeyPoint> keypoints;
	Mat descriptors;
	if(detector == NULL || extractor == NULL || matcher == NULL)
	{
		#ifdef _DEBUG_MODE_WIR
			std::cout<< "CLASS CURRUPTION HAS BEEN DETECTED!!!" << std::endl;
		#endif
		errorCallback(WIRE_GENERAL);
		return -1;
	}
	detector->detect(img, keypoints );
	extractor->compute(img, keypoints, descriptors);
	//Matching
	std::vector< DMatch > matches;
	matcher->match(descriptors,matches);

	int* imgId = new int[trainSamples.size()+5];
	int* classID = NULL;
	if(!imgId)
	{
		#ifdef _DEBUG_MODE_WIR
			std::cout<< " Not Enough Memory " << std::endl; 
		#endif
			errorCallback(WIRE_CANNOT_LOAD_IMAGE);
		return -1; 
	}
	//clearing memory
	for (unsigned int i =0; i< trainSamples.size();i++)
		imgId[i]=0;

	classID = new int[maxClassLabel+1];
	if(!classID)
	{
		#ifdef _DEBUG_MODE_WIR
			std::cout<< " Not Enough Memory " << std::endl; 
		#endif
		errorCallback(WIRE_CANNOT_LOAD_IMAGE);
		return -1;
	}
	//clearing memory
	for (int i =0; i<=maxClassLabel;i++)
	classID[i]=0;

	double max_dist = 0; double min_dist = 100; 
	//-- Quick calculation of max and min distances between keypoints
	  for( int i = 0; i < descriptors.rows; i++ )
	  { double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	  }

#ifdef _DEBUG_MODE_WIR
	  printf("-- Max dist : %f \n", max_dist );
	  printf("-- Min dist : %f \n", min_dist );
#endif
	  //-- select "good" matches (i.e. whose distance is less than 3*min_dist )
	  std::vector< DMatch > good_matches;
	  for( int i = 0; i < descriptors.rows; i++ )
	  { if( matches[i].distance < param.goodSelectionMultilier*min_dist )
		{ good_matches.push_back( matches[i]); }
	  }
#ifdef _DEBUG_MODE_WIR
	  cout<<"% of selected points "<< good_matches.size()/(double)descriptors.rows <<endl;
#endif

	for (unsigned int i =0;i<good_matches.size();i++)
	{
		imgId[good_matches[i].imgIdx]++;
		if ((param.useClassLabel !=0) && (maxClassLabel >0))
			classID[trainSamples[good_matches[i].imgIdx].classLabel]++; // WARNING HAS NOT BEEN TESTED
	}
	//cout << "TEST1" << endl;
	WIRResult tmpResult;
	results.clear();
	if (param.useClassLabel == 0 || maxClassLabel <=0)
		max_matches = MIN(max_matches,trainSamples.size());
	else
		max_matches = MIN(MIN(max_matches,(unsigned int)maxClassLabel),trainSamples.size()); //!!!!
	for (unsigned int j =0; j<max_matches;j++)
	{
		int max_id = 0;
		for (unsigned int i =0; i<trainSamples.size();i++)
		{
			if(imgId[i]>imgId[max_id])
				max_id = i;
			#ifdef _DEBUG_MODE_WIR
				if(imgId[i]!=0) printf("%s %d\n",trainSamples[i].imageName,imgId[i]);
			#endif
		}

		if ((param.useClassLabel !=0) && (maxClassLabel >0))
		{
			int max_class_id = 0;
			for (int i =0; i<=maxClassLabel;i++)
			{
				if(classID[i]>classID[max_class_id])
					max_class_id = i;
				#ifdef _DEBUG_MODE_WIR
					if(classID[i]!=0) printf("Class ID : %d %d\n",i, classID[i]);
				#endif
			}
			classID[max_class_id] = -1;
			tmpResult.classLabel = max_class_id;
		}
		else
			tmpResult.classLabel = 0;

		if(param.useHistProcessing && loadedFromFile==0)
		//Histogram matching (EXPEREMENTAL)
		if (dbKeyPoints.size() == dbDescriptors.size())
		{
			int queryMaxXId = 0; int queryMaxYId = 0;
			int queryMinXId = 0; int queryMinYId = 0;
			int objMaxXId = 0; int objMaxYId = 0;
			int objMinXId = 0; int objMinYId = 0;
			for (size_t i = 0; i<good_matches.size(); i++)
			{
				if(good_matches[i].imgIdx == max_id)
				{
					if((dbKeyPoints[max_id])[good_matches[i].trainIdx].pt.x<
						(dbKeyPoints[max_id])[objMinXId].pt.x)
						objMinXId = good_matches[i].trainIdx;

					if((dbKeyPoints[max_id])[good_matches[i].trainIdx].pt.y<
						(dbKeyPoints[max_id])[objMinYId].pt.y)
						objMinYId = good_matches[i].trainIdx;

					if((dbKeyPoints[max_id])[good_matches[i].trainIdx].pt.x>
						(dbKeyPoints[max_id])[objMaxXId].pt.x)
						objMaxXId = good_matches[i].trainIdx;

					if((dbKeyPoints[max_id])[good_matches[i].trainIdx].pt.y>
						(dbKeyPoints[max_id])[objMaxYId].pt.y)
						objMaxYId = good_matches[i].trainIdx;
					///////////////////////////////////////////////////////

					if(keypoints[good_matches[i].queryIdx].pt.x<
						keypoints[queryMinXId].pt.x)
						queryMinXId = good_matches[i].queryIdx;

					if(keypoints[good_matches[i].queryIdx].pt.y<
						keypoints[queryMinYId].pt.y)
						queryMinYId = good_matches[i].queryIdx;

					if(keypoints[good_matches[i].queryIdx].pt.x>
						keypoints[queryMaxXId].pt.x)
						queryMaxXId = good_matches[i].queryIdx;

					if(keypoints[good_matches[i].queryIdx].pt.y>
						keypoints[queryMaxYId].pt.y)
						queryMaxYId = good_matches[i].queryIdx;
				}
			}
			cv::Rect queryRect, objRect;
			queryRect.x = (int)keypoints[queryMinXId].pt.x; 
			queryRect.y = (int)keypoints[queryMinYId].pt.y;
			queryRect.width = MAX(5, (int)keypoints[queryMaxXId].pt.x - queryRect.x);
			queryRect.height = MAX(5, (int)keypoints[queryMaxYId].pt.y - queryRect.y);
			//-----------------------//
			objRect.x = (int)(dbKeyPoints[max_id])[objMinXId].pt.x; 
			objRect.y = (int)(dbKeyPoints[max_id])[objMinYId].pt.y;
			objRect.width = MAX(5, (int)(dbKeyPoints[max_id])[objMaxXId].pt.x - objRect.x);
			objRect.height = MAX(5, (int)(dbKeyPoints[max_id])[objMaxYId].pt.y - objRect.y);
			//--------------------//
			Mat queryImg = img(queryRect);
			Mat img2 = imread(trainSamples[max_id].imagePath, IMREAD_GRAYSCALE ); //!!! IMREAD_GRAYSCALE
			if(img2.data )
			{ 
				Mat objImg = img2(objRect);
				Mat objHist, queryHist;
				float range[] = { 0, 256 } ; //the upper boundary is exclusive
				const float* histRange = { range };
				bool uniform = true; bool accumulate = false;
				int histSize = 64; //64 bins
				int histChannels = 0; //means 1 channel
				calcHist(&objImg, 1, &histChannels, Mat(), objHist, 1, &histSize, &histRange, uniform, accumulate );
				normalize(objHist, objHist, 0, 1, NORM_MINMAX, -1, Mat());
				calcHist(&queryImg, 1, &histChannels, Mat(), queryHist, 1, &histSize, &histRange, uniform, accumulate );
				normalize(queryHist, queryHist, 0, 1, NORM_MINMAX, -1, Mat());

				tmpResult.hist = compareHist(queryHist, objHist, CV_COMP_BHATTACHARYYA);

				////Homography
				//{
				//	 //-- Localize the object from img_1 in img_2
				//	  std::vector<Point2f> obj;
				//	  std::vector<Point2f> scene;

				//	  for( size_t i = 0; i < good_matches.size(); i++ )
				//	  {
				//		//-- Get the keypoints from the good matches
				//		  if(good_matches[i].imgIdx != max_id)
				//			  continue;
				//		obj.push_back( keypoints[ good_matches[i].queryIdx ].pt );
				//		scene.push_back((dbKeyPoints[max_id])[ good_matches[i].trainIdx ].pt );
				//	  }

				//	  Mat H = findHomography( obj, scene, RANSAC );
				//	  Mat mImg = Mat(Size(img2.size().height*2,img2.size().width*2),CV_8UC1); 
				//	  cv::warpPerspective(img2,mImg,H,mImg.size());
				//	  imshow("k",mImg);
				//	  imshow("original",img);
				//	  waitKey(0);
				//}

				//imshow("objHist", objImg);
				//imshow("queryHist", queryImg);
			}; 
		}
		tmpResult.fileName[0] = 0;
		tmpResult.filePath[0] = 0;
		tmpResult.year = detectedYear;
		strcpy(tmpResult.fileName, trainSamples[max_id].imageName);
		strcpy(tmpResult.filePath, trainSamples[max_id].imagePath);
		tmpResult.assignedClassLabel = trainSamples[max_id].classLabel;
		tmpResult.propobility = 0.0;
#ifdef _DEBUG_MODE_WIR
		cout<<"Max_ID: "<<max_id <<" Count: "<<imgId[max_id]<<endl;
#endif
		imgId[max_id] = -1;
		results.push_back(tmpResult);
	} 
	delete[] imgId;
	  delete[] classID;
	  return results.size();
};

int WIR01::ExtractDescriptors(const char* file_path, Mat& descriptors, vector<KeyPoint>& keypoints)
{
	Mat img = imread(file_path, IMREAD_GRAYSCALE ); //!!! IMREAD_GRAYSCALE
	if(!img.data )
	{ 
		#ifdef _DEBUG_MODE_WIR
			std::cout<< " --("<<file_path<<") Error reading images " << std::endl;
			errorCallback(WIRE_CANNOT_LOAD_IMAGE);
		#endif		
		return 0; 
	};
	ImagePreProcessing (img);
	keypoints.clear();
	if(detector == NULL || extractor == NULL)
	{
		#ifdef _DEBUG_MODE_WIR
			std::cout<< "CLASS CURRUPTION HAS BEEN DETECTED!!!" << std::endl;
		#endif
		errorCallback(WIRE_GENERAL);
		return -1;
	}
	detector->detect(img, keypoints );
	if (keypoints.size() <= 0)
		return 0;
	extractor->compute(img, keypoints, descriptors);
	return 1;
};

//addTrainSamples; returs number of added samples
int WIR01::addTrainSamples(vector<WIRTrainSample>& samples)
{
	if(samples.size() == 0)
		return 0;
	int addedCount = 0;
	Mat tmpDescriptor;
	vector<KeyPoint> tmpKeyPoints;
	for (unsigned int i = 0; i<samples.size(); i++)
	{	
		#ifdef _DEBUG_MODE_WIR
				cout << samples[i].imagePath << endl;
		#endif
		if(ExtractDescriptors(samples[i].imagePath,tmpDescriptor,tmpKeyPoints))
		{
			trainSamples.push_back(samples[i]);
			dbDescriptors.push_back(tmpDescriptor);
			dbKeyPoints.push_back(tmpKeyPoints);
			addedCount++;
			//finding the biggest class label;
			if(samples[i].classLabel > maxClassLabel)
				maxClassLabel = samples[i].classLabel;
		}
	}
	if (addedCount >0)
	{
		train();
	}
	return addedCount;

};

void WIR01::setRecognitionParam(WIRParam param)
{
	this->param = param;
	if(detector != NULL)
	{delete detector; detector = NULL;};
	if (strcmp(param.detectorType, "SURF")==0)
		detector = 	new SurfFeatureDetector(param.threshold);
	else
		if (strcmp(param.detectorType, "FAST")==0)
			detector = 	new FastFeatureDetector((int)param.threshold);
		else
			detector = 	FeatureDetector::create(param.detectorType);
	if (detector == NULL)
		WIRInternalPanic(WIRE_NOT_ENOUGH_MEMORY);

	if (extractor != NULL)
	{delete extractor; extractor = NULL;};
	if (strcmp(param.descriptorExtractorType, "SURF")==0)
		if (strcmp(param.detectorType, "SURF")==0)
		extractor = new SurfDescriptorExtractor(((SurfFeatureDetector*)detector)->hessianThreshold, ((SurfFeatureDetector*)detector)->nOctaves, 
			((SurfFeatureDetector*)detector)->nOctaveLayers,
			((SurfFeatureDetector*)detector)->extended, ((SurfFeatureDetector*)detector)->upright);
		else
			extractor = new SurfDescriptorExtractor(param.threshold);
	else
		if (strcmp(param.descriptorExtractorType, "BRIEF")==0)
			extractor = new BriefDescriptorExtractor(BRIEF_DECTRIPTOR_SIZE);
		else
			extractor = DescriptorExtractor::create(param.descriptorExtractorType);
	if (extractor == NULL)
		WIRInternalPanic(WIRE_NOT_ENOUGH_MEMORY);

	if (matcher != NULL)
	{delete matcher; matcher = NULL;};
	if (strcmp(param.descriptorExtractorType, "BRIEF")==0)
		matcher = new FlannBasedMatcher(new cv::flann::LshIndexParams(12, 20, 2), new cv::flann::SearchParams());
	else
		matcher = new FlannBasedMatcher();
	if (matcher == NULL)
		WIRInternalPanic(WIRE_NOT_ENOUGH_MEMORY);
	loadOCRParam();
	ocr.setLabelExtration(param.labelExtraction);
	dbDescriptors.clear();
	GetDescriptors();
};

int WIR01::saveTrainingDB(const char* file_path)
{
	FileStorage fs(file_path, FileStorage::WRITE);
	if(!fs.isOpened())
	{
	#ifdef _DEBUG_MODE_WIR
		cout<< "Cannot save settings to "<<file_path<<endl;
		WIRInternalPanic(WIRE_CANNOT_PROCESS_IO);
	#endif
		return -1;
	}
	string tmpstr("FULL");
	fs<<"StorageParam"<<tmpstr;
	fs<<"Param"<<param;
	fs<<"MaxClassLabel"<<maxClassLabel;
	fs<<"Descriptors"<<dbDescriptors;
	fs<<"TrainSamples"<<"[";
	for(unsigned int i =0; i<trainSamples.size();i++)
		fs<<trainSamples[i];
	fs<<"]";
	fs.release();
	return 1;
};


//Save data in banch of files
int WIR01::saveTrainingDBPartially(const vector<const char*>& directories, unsigned int filesPerDir, unsigned int descriptorsPerFile, const char* baseFileName)
{
	if (baseFileName == NULL)
	{
#ifdef _DEBUG_MODE_WIR
		std::cout<<"Cannot use empty baseFileName" << endl;
#endif
		WIRInternalPanic(WIRE_CANNOT_PROCESS_IO);
		return -1;
	}
	if (directories.size() == 0)
	{
#ifdef _DEBUG_MODE_WIR
		std::cout<<"Emptry DIR path" << endl;
#endif
		WIRInternalPanic(WIRE_CANNOT_PROCESS_IO);
		return -1;
	}
	if (directories[0] == NULL)
	{
#ifdef _DEBUG_MODE_WIR
		std::cout<<"Emptry DIR path" << endl;
#endif
		WIRInternalPanic(WIRE_CANNOT_PROCESS_IO);
		return -1;
	}
	FileStorage fs;
	const string partial_string("PARTIAL");
	unsigned int elementID = 0;
	unsigned int storedFilesCount = 0;
	unsigned int currentDirectory = 0;
	filesPerDir = filesPerDir<1?1:filesPerDir;
	descriptorsPerFile = descriptorsPerFile<10?10:descriptorsPerFile;
	char pathName[2048];
	char buffer[10];
	
	while(elementID<= dbDescriptors.size() -1)
	{
		pathName[0] = 0;
		buffer[0] = 0;
		if(storedFilesCount >= filesPerDir && currentDirectory != directories.size()-1)
		{
			currentDirectory++;
			storedFilesCount = 0;
		};
		strcpy(pathName,directories[currentDirectory]);
		strcat(pathName,"\\");
		strcat(pathName,baseFileName);
		sprintf(buffer,"%d", storedFilesCount+currentDirectory*filesPerDir);
		strcat(pathName,buffer);
		strcat(pathName,".xml");
		fs.open(pathName,FileStorage::WRITE);
		if(!fs.isOpened())
		{
		#ifdef _DEBUG_MODE_WIR
			std::cout<< "Cannot save settings to "<<pathName<<endl;
		#endif
			if (currentDirectory == directories.size()-1)
			{
				WIRInternalPanic(WIRE_CANNOT_PROCESS_IO);
				return -1;
			}
			currentDirectory++;
			continue;
		}

		//////////////////////////////////
		////SELECTING FILES TO STORE//////
		//////////////////////////////////
		vector<cv::Mat> tmpDB;
		vector<WIRTrainSample> tmpTrainSamples;
		for(size_t i = elementID; i< MIN(elementID+descriptorsPerFile, dbDescriptors.size()-1); i++)
		{
			tmpDB.push_back(dbDescriptors[i]);
			tmpTrainSamples.push_back(trainSamples[i]);
		}
		fs<<"StorageParam"<<partial_string;
		fs<<"Param"<<param;
		fs<<"MaxClassLabel"<<maxClassLabel;
		fs<<"Descriptors"<<tmpDB;
		fs<<"TrainSamples"<<"[";
		for(unsigned int i =0; i<tmpTrainSamples.size();i++)
			fs<<tmpTrainSamples[i];
		fs<<"]";
		fs.release();
		
		elementID+=descriptorsPerFile;
		storedFilesCount++;
	}
	return 1;
}

int WIR01::SaveBinary(const char* directory)
{
	if (directory == NULL)
	{
#ifdef _DEBUG_MODE_WIR
		std::cout<<"Cannot use empty baseFileName" << endl;
#endif
		WIRInternalPanic(WIRE_CANNOT_PROCESS_IO);
		return -1;
	}
	char pathName[2048];
	pathName[0] = 0;
	strcpy(pathName,directory);
	strcat(pathName,"\\");
	strcat(pathName,"WIRSettings.xml");
	FileStorage fs(pathName,FileStorage::WRITE);

	if(!fs.isOpened())
		{
		#ifdef _DEBUG_MODE_WIR
			std::cout<< "Cannot save settings to "<<pathName<<endl;
		#endif
				WIRInternalPanic(WIRE_CANNOT_PROCESS_IO);
				return -1;
		}
	 fs<<"StorageParam"<<"BINARY_FULL";
	fs<<"Param"<<param;
	fs<<"MaxClassLabel"<<maxClassLabel;
	fs<<"TrainSamples"<<"[";
	for(unsigned int i =0; i<trainSamples.size();i++)
		fs<<trainSamples[i];
	fs<<"]";
	fs.release();

	for(size_t elementID = 0; elementID<dbDescriptors.size(); elementID++)
	{
		pathName[0] = 0;
		strcpy(pathName,directory);
		strcat(pathName,"\\");
		strcat(pathName,trainSamples[elementID].imageName);
		strcat(pathName,".pgm");
		

		//////////////////////////////////
		////SELECTING FILES TO STORE//////
		//////////////////////////////////
		//if(!dbDescriptors[elementID].data)
		//	cout<<"Saving!!!"<<pathName<<endl;
		if(!imwrite(pathName, dbDescriptors[elementID]))
		{
#ifdef _DEBUG_MODE_WIR
			std::cout<< "Cannot save "<<pathName<<endl;
#endif
				WIRInternalPanic(WIRE_CANNOT_PROCESS_IO);
				return -1;
		};
	}
	return 1;
}
int WIR01::LoadBinary(const char* directory)
{
	if (directory == NULL)
	{
#ifdef _DEBUG_MODE_WIR
		std::cout<<"Cannot use empty baseFileName" << endl;
#endif
		WIRInternalPanic(WIRE_CANNOT_PROCESS_IO);
		return -1;
	}
	char pathName[2048];
	pathName[0] = 0;
	strcpy(pathName,directory);
	strcat(pathName,"\\");
	strcat(pathName,"WIRSettings.xml");
	FileStorage fs(pathName,FileStorage::READ);

	if(!fs.isOpened())
		{
		#ifdef _DEBUG_MODE_WIR
			std::cout<< "Cannot load settings to "<<pathName<<endl;
		#endif
				WIRInternalPanic(WIRE_CANNOT_PROCESS_IO);
				return -1;
		}

	string tmpstr;
	fs["StorageParam"]>>tmpstr;
	if(tmpstr == "BINARY_FULL")
	{
		dbDescriptors.clear();
		trainSamples.clear();
		loadedFromFile = true;
		WIRParam tmpParam;
		fs["Param"]>>tmpParam;
		setRecognitionParam(tmpParam);
		fs["MaxClassLable"]>>maxClassLabel;
		fs["TrainSamples"]>>trainSamples;

		for (size_t i =0; i<trainSamples.size();i++)
		{
			pathName[0] = 0;
			strcpy(pathName,directory);
			strcat(pathName,"\\");
			strcat(pathName,trainSamples[i].imageName);
			strcat(pathName,".pgm");

			Mat img = imread(pathName,-1);
			if(!img.data )
				if(img.rows <0 && img.cols < 0)
				{ 
					#ifdef _DEBUG_MODE_WIR
					std::cout<< " --("<<pathName<<") Error reading images " << std::endl;
					WIRInternalPanic(WIRE_GENERAL);
					#endif		
					return -1; 
				};
			dbDescriptors.push_back(img);
		};
		train();
	}
	return dbDescriptors.size();
}

int WIR01::loadTrainingDB(const char* file_path)
{
	
	FileStorage fs(file_path, FileStorage::READ);
	if(!fs.isOpened())
	{
	#ifdef _DEBUG_MODE_WIR
		cout<< "Cannot load settings from "<<file_path<<endl;
		WIRInternalPanic(WIRE_CANNOT_PROCESS_IO);
	#endif
		return -1;
	}
	string tmpstr;
	fs["StorageParam"]>>tmpstr;
	WIRParam tmpParam;
	fs["Param"]>>tmpParam;
	if(tmpstr == "PARTIAL"&& loadedFromFile)
	{
		vector<cv::Mat> tmpDB;
		vector<WIRTrainSample> tmpTrainSample;
		int tmpMaxClassLabel;
		fs["MaxClassLable"]>>tmpMaxClassLabel;
		fs["Descriptors"]>>tmpDB;
		fs["TrainSamples"]>>tmpTrainSample;
		if(tmpTrainSample.size() != tmpDB.size())
		{
			fs.release();
			return 0;
		}
		maxClassLabel = MAX(maxClassLabel, tmpMaxClassLabel);
		for (size_t i =0; i<tmpTrainSample.size();i++)
		{
			dbDescriptors.push_back(tmpDB[i]);
			trainSamples.push_back(tmpTrainSample[i]);
		};
		matcher->add(tmpDB);
		matcher->train(); //?
	}
	else
	{
		//!!! TO DO: modify
		dbDescriptors.clear();
		trainSamples.clear();
		loadedFromFile = true;
		WIRParam tmpParam;
		fs["Param"]>>tmpParam;
		setRecognitionParam(tmpParam);
		fs["MaxClassLable"]>>maxClassLabel;
		fs["Descriptors"]>>dbDescriptors;
		fs["TrainSamples"]>>trainSamples;
		//
		train();
		loadOCRParam();
	}
	fs.release();
	return 1;
};

void WIR01::train(void)
{
	if(matcher == NULL)
	{
		#ifdef _DEBUG_MODE_WIR
			std::cout<< "CLASS CURRUPTION HAS BEEN DETECTED!!!" << std::endl;
		#endif
		WIRInternalPanic(WIRE_GENERAL);
	}
	else
	{
		matcher->clear();
		matcher->add(dbDescriptors);
		matcher->train();
	}
};

int WIR01::GetDescriptors()
{
	if(trainSamples.size()>0)
	{
		vector<WIRTrainSample> tmpSampels = trainSamples;
		trainSamples.clear();
		return addTrainSamples(tmpSampels);
	}
	else
		return 0;
};

void WIR01::ImagePreProcessing( Mat& image)
{
	//do nothing
};

void WIR01::clear(void)
{
	dbDescriptors.clear();
	trainSamples.clear();
	maxClassLabel = 0;
	matcher->clear();
	loadedFromFile = false;
}

bool WIR01::GenerateUpdates(const WIRParam params, vector<WIRTrainSample>& samples, vector<const char*> directories, 
		unsigned int filesPerDir, unsigned int descriptorsPerFile, const char* baseFileName)
{
	if(samples.size() == 0 || directories.size() == 0)
		return false;
	WIR01 agent(params);
	if(agent.addTrainSamples(samples))
		return agent.saveTrainingDBPartially(directories,filesPerDir,descriptorsPerFile,baseFileName)>0?true:false;
	return false;
}