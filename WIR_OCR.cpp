#include "WIR_OCR.h"


WIR_OCR::WIR_OCR(void):trainData(classes * train_samples,ImageSize, CV_32FC1),
	trainClasses(classes * train_samples, 1, CV_32FC1),maxWidthF(0.25), maxHeightF(0.15)
{
	knearest = new KNearest();
	errorCallback = NULL;
	showDebugInformation = 0;
	initialized = 0;
	runAdditionalDilation = 0;
	labelExtraction = WIR_EL_SOFT;
}


WIR_OCR::~WIR_OCR(void)
{
#ifndef WIR_DO_NOT_DESTROY_KNEAREST
	try
	{
		delete knearest;
	}
	catch(exception e)
	{
		cout<<"Knearest Error!" <<endl;
	}  
#endif // !WIR_DO_NOT_DESTROY_KNEAREST

}


bool compareObj(const std::vector<cv::Point2i > &objA,  const std::vector<cv::Point2i >& objB)
{
	return objA.size()<objB.size();
}
bool compareObjByX(const cv::Rect &objA, const  cv::Rect& objB)
{
	return objA.x<objB.x;
}
void WIR_OCR::setErrorCallback(WIRErrorCallback errorCallback)
{
	this->errorCallback = errorCallback;
}

//Function for testing
int WIR_OCR::m()
{

	showDebugInformation = 1;
	vector <unsigned int> recognizedYears;
	LearnFromImages(trainData, trainClasses,"./Imgs");

	RunSelfTest(*knearest, "./Imgs");

	cout << "Analysis\n";
	Mat  //image = imread("101182h.jpg", CV_LOAD_IMAGE_GRAYSCALE);
 //image = imread("62319.jpg", CV_LOAD_IMAGE_GRAYSCALE);
 //image = imread("151208.jpg", CV_LOAD_IMAGE_GRAYSCALE);
 //image = imread("246489.jpg", CV_LOAD_IMAGE_GRAYSCALE);
 image = imread("123871h.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	AnalyseImage(image, recognizedYears);
	for (size_t i = 0; i <recognizedYears.size(); i++)
		cout<<recognizedYears[i]<<endl;
	showDebugInformation = 0;
	return 0;

}

void WIR_OCR::PreProcessImage(Mat *inImage,Mat *outImage,int sizex, int sizey)
{
 Mat grayImage,blurredImage,thresholdImage,contourImage,regionOfInterest;

 vector<Rect> contours;
 std::vector < std::vector<cv::Point2i > > blobs;

 int morph_size = 6;
  Mat element = getStructuringElement(MORPH_RECT , Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

  /// Apply the specified morphology operation
  morphologyEx( *inImage, blurredImage, 6, element );
 //adaptiveThreshold(blur, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 0);
 double thOtsu = threshold(blurredImage,thresholdImage,128,255,CV_THRESH_BINARY | CV_THRESH_OTSU);

 thresholdImage.copyTo(contourImage);

 //добавить извлечение объекта
 FindBlobs(contourImage,blobs);
 int maxArea = 0; int maxID = 0;
 if(FindObjects(blobs,contourImage.size(),contours))
 {
	 for(size_t i = 0; i<contours.size(); i++)
	 {
		 int tmpArea = contours[i].height * contours[i].width;
		 if (maxArea<tmpArea)
		 {
			 maxArea = tmpArea;
			 maxID = i;
		 }
	 }
	 regionOfInterest = thresholdImage(contours[maxID]);
	 
 }
 else
	 thresholdImage.copyTo(regionOfInterest);

	 cv::resize(regionOfInterest,*outImage, Size(sizex, sizey));

}

//int WIR_OCR::LearnFromImages(CvMat* trainData, CvMat* trainClasses)
//{
// Mat img;
// char file[255];
// for (int i = 0; i < classes; i++)
// {
//  sprintf(file, "%s/%d.png", pathToImages, i);
//  img = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
//  if (!img.data)
//  {
//    cout << "File " << file << " not found\n";
//    return -1;
//  }
//  Mat outfile;
//  if(img.type() == CV_8UC1)
// PreProcessImage(&img, &outfile, sizex, sizey);
//  else
//	  printf("!!!!");
//  for (int n = 0; n < ImageSize; n++)
//  {
//   trainData->data.fl[i * ImageSize + n] = outfile.data[n];
//  }
//  trainClasses->data.fl[i] = (float)i;
// }
// return 1;
//}

int WIR_OCR::LearnFromImages(Mat& trainData, Mat& trainClasses, const char* pathToImages)
{
 Mat img;
 char file[255];
 for (int i = 0; i < classes; i++)
 {
  sprintf(file, "%s/%d.png", pathToImages, i);
  img = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
  if (!img.data)
  {
    cout << "File " << file << " not found\n";
    return -1;
  }
  Mat outfile;
  if(img.type() == CV_8UC1)
 PreProcessImage(&img, &outfile, sizex, sizey);
  else
	  printf("!!!!");
  for (int n = 0; n < ImageSize; n++)
  {
	  trainData.at<float>(i,n) = (float)outfile.data[n];
  }
  trainClasses.at<float>(i) = (float)i;
 }

 for (int i = 0; i < classes; i++)
 {
  sprintf(file, "%s/%dk.png", pathToImages, i);
  img = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
  if (!img.data)
  {
    cout << "File " << file << " not found\n";
    return -1;
  }
  Mat outfile;
  if(img.type() == CV_8UC1)
 PreProcessImage(&img, &outfile, sizex, sizey);
  else
	  printf("!!!!");
  for (int n = 0; n < ImageSize; n++)
  {
	  trainData.at<float>(classes+i,n) = (float)outfile.data[n];
  }
  trainClasses.at<float>(classes+i) = (float)i;
 }

  for (int i = 0; i < classes; i++)
 {
  sprintf(file, "%s/%dq.png", pathToImages, i);
  img = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
  if (!img.data)
  {
    cout << "File " << file << " not found\n";
    return -1;
  }
  Mat outfile;
  if(img.type() == CV_8UC1)
 PreProcessImage(&img, &outfile, sizex, sizey);
  else
	  printf("!!!!");
  for (int n = 0; n < ImageSize; n++)
  {
	  trainData.at<float>(2*classes+i,n) = (float)outfile.data[n];
  }
  trainClasses.at<float>(2*classes+i) = (float)i;
 }
  initialized = knearest->train(trainData, trainClasses)?1:0;
  return initialized;
}

void WIR_OCR::RunSelfTest(KNearest& knn2, const char* pathToImages)
{
 Mat img;
 CvMat* sample2 = cvCreateMat(1, ImageSize, CV_32FC1);
 // SelfTest
 char file[255];
 int z = 0;
 while (z++ < 10)
 {
  int iSecret = rand() % 10;
  sprintf(file, "%s/%d.png", pathToImages, iSecret);
  img = imread(file, 0);
  Mat stagedImage;
  PreProcessImage(&img, &stagedImage, sizex, sizey);
  for (int n = 0; n < ImageSize; n++)
  {
   sample2->data.fl[n] = stagedImage.data[n];
  }
  float detectedClass = knn2.find_nearest(sample2, 1);
  if (iSecret != (int) ((detectedClass)))
  {
   cout << "Digit " << iSecret << " recognized as"
     << (int) ((detectedClass));
   exit(1);
  }
  cout << "Tested" << (int) ((detectedClass)) << "\n";
 }
 cvReleaseMat(&sample2);
}

int WIR_OCR::AnalyseImage(const char* image_path, vector<unsigned int>& recognizedYears,cv::Rect* wineLabel)
{
	if(image_path == NULL)
		{ 
		#ifdef _DEBUG_MODE_WIR_OCR
			std::cout<< " --(NOT DEFFINED) Error reading images " << std::endl;
			WIRInternalPanic(WIRE_CANNOT_LOAD_IMAGE);
		#endif		
		return -1; 
	};
	if(image_path[0] == 0)
		{ 
		#ifdef _DEBUG_MODE_WIR_OCR
			std::cout<< " --("<<image_path<<") Error reading images " << std::endl;
			WIRInternalPanic(WIRE_CANNOT_LOAD_IMAGE);
		#endif		
		return -1; 
	};
	Mat img = imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
  if (!img.data)
  {
#ifdef _DEBUG_MODE_WIR_OCR
	  cout << "File " << image_path << " not found\n";
#endif
	WIRInternalPanic(WIRE_CANNOT_LOAD_IMAGE);
    return -1;
  }
  return AnalyseImage(img, recognizedYears,wineLabel);
}

//Returns 1 if currepted
int WIR_OCR::InternalCurruptionCheck()
{
	if(knearest == NULL)
		return 1;
	return !(trainData.size().height>0 && trainData.size().height==trainClasses.size().height);
}
bool WIR_OCR::DetectLable(const cv::Mat& inputImage, cv::Rect& wineLabel, double xScale, double yScale, double sizeScale)
{
	Mat image;
	Mat gray, blur, thresh;
	//Data correction;
	xScale = abs(xScale);
	yScale = abs(yScale);
	sizeScale = abs(sizeScale);
	sizeScale = sizeScale < 0.05 ? 0.05 : sizeScale;
	sizeScale = sizeScale > 0.5 ? 0.25 : sizeScale;
	xScale = MAX(sizeScale, xScale);
	xScale = sizeScale + xScale > 1 ? 1-sizeScale : xScale;
	yScale = MAX(sizeScale, yScale);
	yScale = sizeScale + yScale > 1 ? 1-sizeScale : yScale;


	std::vector < cv::Rect > contours;
	std::vector < std::vector<cv::Point2i > > blobs;
	if(inputImage.channels()>1)
		inputImage.convertTo(image,CV_BGR2GRAY);
	else
		inputImage.copyTo(image);

	wineLabel.x = 0;
	wineLabel.y = 0;
	wineLabel.height = inputImage.size().height;
	wineLabel.width = inputImage.size().width;

	int morph_size2 = (int)std::floor(MIN(image.size().height,image.size().width)/sizeMorphElement1);
	int morph_size3 = 3;

	Mat element2 = getStructuringElement(MORPH_RECT , Size( 2*morph_size2 + 1, 2*morph_size2+1 ), Point( morph_size2, morph_size2 ) );
	Mat element3 = getStructuringElement(MORPH_RECT , Size( 2*morph_size3+1, 2*morph_size3+1 ), Point( morph_size3, morph_size3 ) );
	//Detecting background color
	Mat img = image(cv::Rect((int)image.size().width*(xScale-sizeScale), (int)image.size().height*(yScale-sizeScale), 
		(int)image.size().width*2*sizeScale, (int)image.size().height*2*sizeScale));
	double threshold = cv::threshold(img,gray,128,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
	double grayLavel = floor(cv::mean(gray)[0]);
	cv::threshold(image,gray,threshold,255,CV_THRESH_BINARY);
	if(grayLavel<darkBackgoundSeparater)
	{
		gray = 255 - gray;
		//Label Extraction
		//Extracting light elements
		cv::morphologyEx( image, blur, MORPH_TOPHAT, element3 );

	}
	else
	{
		//Light background
		//LabelExtraction
		//Extracting dark elements
		cv::morphologyEx( image, blur, MORPH_BLACKHAT, element3 );
	}

	cv::threshold(blur,blur,125,255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//Bluring the label
	FindBlobs(gray,blobs);
	int maxBlolbIndex = FindMaxBlob(blobs);
	//Extracting the blured label
	gray = Mat::zeros(gray.size(), gray.type());
	if (maxBlolbIndex>=0)
		RegionColorize(gray,blobs[maxBlolbIndex]);
	cv::bitwise_or(blur,gray,gray);
	//blude and detect again
	cv::morphologyEx(gray,thresh,MORPH_CLOSE,element2);
	FindBlobs(thresh,blobs);
	maxBlolbIndex = FindMaxBlob(blobs);
	if (maxBlolbIndex>=0)
			wineLabel = GetBlobRect(blobs[maxBlolbIndex]);
	return true;
}
int WIR_OCR::AnalyseImage(const Mat& image2, vector<unsigned int>& recognizedYears, cv::Rect* wineLabel)
{
	Mat image;
	if(!initialized)
		return WIR_OCR_NOT_INITIALIZED;
	if(knearest == NULL)
	{
#ifdef _DEBUG_MODE_WIR_OCR
		cout<<"KNUL"<<endl;
#endif
		WIRInternalPanic(WIRE_NOT_ENOUGH_MEMORY);
		return -1;
	}
	if(InternalCurruptionCheck())
	{
#ifdef _DEBUG_MODE_WIR_OCR
		cout<<"OCR has been currepted. Plz. resetup it"<<endl;
#endif
		WIRInternalPanic();
		return -1;
	}
	
	recognizedYears.clear();
	//Checking color state;
	if(image2.channels()>1)
		image2.convertTo(image,CV_BGR2GRAY);
	else
		image2.copyTo(image);
#ifdef _DEBUG_MODE_WIR_OCR
		cout<<"Copied"<<endl;
#endif

		if (wineLabel != NULL)
		{
			DetectLable(image,*wineLabel);
			if(labelExtraction != WIR_EL_NONE)
				image = image(*wineLabel);
		};
 int maxX, maxY, minX, minY;
 Mat gray, blur, thresh;
 std::vector < cv::Rect > contours;
 std::vector < std::vector<cv::Point2i > > blobs;

	//forming morphology elements
	int morph_size = (int)std::floor(MIN(image.size().height,image.size().width)/sizeMorphElement1);
	int morph_size2 = (int)std::floor(MIN(image.size().height,image.size().width)/sizeMorphElement2);
if(showDebugInformation)
	std::cout<<"morh_size : " <<morph_size<<endl;
	Mat element = getStructuringElement(MORPH_RECT , Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
	Mat element2 = getStructuringElement(MORPH_RECT , Size( 2*morph_size2 + 1, 2*morph_size2+1 ), Point( morph_size2, morph_size2 ) );
	cv::threshold(image,gray,128,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
	if(floor(cv::mean(gray)[0])<darkBackgoundSeparater)
	{
		cv::morphologyEx( image, blur, MORPH_TOPHAT, element );
	}
	else
	{
		cv::morphologyEx( image, blur, MORPH_BLACKHAT, element );
	}
	if(showDebugInformation)
		cv::imshow("T or B transform", blur);
//Thresholding using Otsu method
 cv::threshold(blur,thresh,128,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
 if(showDebugInformation)
		cv::imshow("Thresholding 1", thresh);
 //Connecting internal regions
 cv::morphologyEx(thresh,blur,MORPH_CLOSE,element2);
 if(showDebugInformation)
	cv::imshow("After Morphology filtretion",blur);
 #ifdef _DEBUG_MODE_WIR_OCR
		cout<<"After Morphology filtretion"<<endl;
#endif
 //findg connected components
    cv::Mat output = cv::Mat::zeros(thresh.size(), CV_8UC1);

#ifdef _DEBUG_MODE_WIR_OCR
		cout<<"Large elements has been removed"<<endl;
#endif

    FindBlobs(blur, blobs);

#ifdef _DEBUG_MODE_WIR_OCR
		cout<<"BLOBS HAS BEEN FOUND"<<endl;
		cout<<"Count : "<<blobs.size()<<endl;
#endif
	//sorting Blobs by size
	std::sort(blobs.begin(),blobs.end(),compareObj);
#ifdef _DEBUG_MODE_WIR_OCR
		cout<<"SORTED"<<endl;
#endif
	//removing the largest objests (top 1% or larger 1% of totall area)
	size_t sizeOfBlobs = (blobs.size()*99)/100;
	unsigned int totalArea = thresh.size().area();
	if(blobs.size()>=WIR_OCR_MIN_BLOBS)
		while(blobs[blobs.size()-1].size()>=totalArea/100 || blobs.size()> sizeOfBlobs)
		{
			blobs.pop_back();
			// Any guess is way way better that no guess at all
			if (blobs.size() <= WIR_OCR_MIN_BLOBS)
				break;
		}
	else
	{
			return -1;
	};
#ifdef _DEBUG_MODE_WIR_OCR
		cout<<"Removed."<<endl;
		cout<<"Count : "<<blobs.size()<<endl;
#endif
	// Randomy color the blobs and select soft region
	RegionColorize(thresh.size(), output, blobs, maxX, minX, maxY, minY);
	if(showDebugInformation)
		cv::imshow("Large elements has been removed",output);
#ifdef _DEBUG_MODE_WIR_OCR
		cout<<"Large elements has been removed"<<endl;
#endif
	cv::Mat output2;
	cv::morphologyEx(output,output2,MORPH_CLOSE,element);
	if(runAdditionalDilation)
		cv::morphologyEx(output2,output2,MORPH_DILATE,element2);
	if(showDebugInformation)
		cv::imshow("Regions prepared",output2);
	blobs.clear();
	output = cv::Mat::zeros(thresh.size(), CV_8UC1);
	FindBlobs(output2, blobs);

	//sorting Blobs by size
	std::sort(blobs.begin(),blobs.end(),compareObj);
	//removing the largest objests (larger that 2% of totall area)
	totalArea = thresh.size().area();
	//cout<<totalArea<<endl;
	if(blobs.size()>2)
		while(blobs[blobs.size()-1].size()>=totalArea/50)
		{
			//cout<<blobs[blobs.size()-1].size()/100<<endl;
			blobs.pop_back();
			if (blobs.size() == WIR_OCR_MIN_BLOBS)
				break;
		}
	else
		return -1;
	//!!! Important constants
	int maxWidth =(int) (0.29 * thresh.size().width);
	int maxHeight =(int) (0.15 * thresh.size().height);
	int tmpMaxX =0, tmpMaxY =0;
	// Randomy color the blobs
    for(size_t i=0; i < blobs.size(); i++) {
		if(blobs[i].size()<10)
			continue;
        //deleting really long objects
		int maxX = 0; int minX = thresh.size().width;
		int maxY = 0; int minY = thresh.size().height;
		for(size_t j=0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;
			if(x<minX) minX = x;
			if(x>maxX) maxX = x;
			if(y<minY) minY = y;
			if(y>maxY) maxY = y;
        }
		//если больше то удаляем
		if(abs(maxY - minY)>maxHeight)
			continue;
		if(abs(maxX - minX)>maxWidth)
			continue;
		// выделяем региона интереса
        for(size_t j=0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;
			output.at<unsigned char>(y,x) = 255;
        }
    }
	if(showDebugInformation)
		cv::imshow("Selected regions",output);
	output2 = cv::Mat::zeros(thresh.size(), CV_8UC1);
	cv::bitwise_and(thresh,output,output2);
	if(showDebugInformation)
	{
		cv::imshow("labelled", output2);
		cv::waitKey(0);
	}

	///////////////////////////////////
	/////DETECTING YEAR///////////////
	/////////////////////////////////
#ifdef _DEBUG_MODE_WIR_OCR
		cout<<"Detecting year"<<endl;
#endif

	CvMat* sample2 = cvCreateMat(1, ImageSize, CV_32FC1);
 	blobs.clear();
	FindBlobs(output2, blobs);
	FindObjects(blobs,output2.size(),contours);
	std::sort(contours.begin(), contours.end(),compareObjByX);
#ifdef _DEBUG_MODE_WIR_OCR
		cout<<"Selected blobs : "<<contours.size()<<endl;
#endif
	//Recognition of all contours
	vector<int> yOfDetectedObjects;
	vector<RecognizedRegion> detectedNumbers;
	for (size_t i = 0; i < contours.size(); i++)
	{
		Mat roi = thresh(contours[i]);
		Mat stagedImage;
		resize(roi,stagedImage, Size(sizex, sizey));
		for (int n = 0; n < ImageSize; n++)
		{
			sample2->data.fl[n] = (float)stagedImage.data[n];
		}

		float result = 1888;
		if (knearest!=NULL)
			result = knearest->find_nearest(sample2, 1);
		else
		{
#ifdef _DEBUG_MODE_WIR_OCR
			cout<<"KNULL2"<<endl;
#endif
			WIRInternalPanic(WIRE_NOT_ENOUGH_MEMORY);
		}
		RecognizedRegion tmpRR(contours[i],(unsigned int)result);
#ifdef _DEBUG_MODE_WIR_OCR
			cout<<(unsigned int)result<<endl;
#endif
		detectedNumbers.push_back(tmpRR);
		yOfDetectedObjects.push_back(contours[i].y);
   }
	
	vector<string> recognizedStrings;
	for (size_t i=0; i<detectedNumbers.size();i++)
	{
		unsigned int isMarkChanged = 0;
		int y = yOfDetectedObjects[i];
		string tmpStr;
		isMarkChanged = 0;
		for (size_t j=0; j<detectedNumbers.size(); j++)
		{
			if(detectedNumbers[j].rect.y<=y &&detectedNumbers[j].rect.y+detectedNumbers[j].rect.height>=y)
			{
				if(detectedNumbers[j].group == 0)
				{
					detectedNumbers[j].group = 1;
					isMarkChanged = 1; //count of changes
				}
				tmpStr+=(unsigned char)(detectedNumbers[j].digit+48);
			}
		}
		//обновилась ли у нас дата.
		if(isMarkChanged)
		//коррекция даты
		switch(tmpStr.length())
		{
			case 3:
				if(tmpStr[0]=='0')
					tmpStr = '2' + tmpStr;
				else
					tmpStr = '1' + tmpStr;
				recognizedStrings.push_back(tmpStr);
			break;
			case 4:
				if(tmpStr[1] == '0')
					tmpStr[0] = '2';
				else
					tmpStr[0] = '1';
				recognizedStrings.push_back(tmpStr);
			break;
		};
	}

#ifdef _DEBUG_MODE_WIR_OCR
	if(recognizedStrings.size()>0)
		cout<<"Recognized years:"<<endl;
	else
		cout<<"NO YEAR has been detected"<<endl;
#endif

	for (size_t i=0; i<recognizedStrings.size();i++)
	{
		recognizedYears.push_back(atoi(recognizedStrings[i].c_str()));
#ifdef _DEBUG_MODE_WIR_OCR
		cout<<recognizedStrings[i]<<endl;
#endif
	}

 cvReleaseMat(&sample2);
 return recognizedYears.size();
}
unsigned int WIR_OCR::AnalyseImage(const Mat& image, cv::Rect* wineLabel)
{
	vector<unsigned int> recognizedYears;
	if (wineLabel != NULL)
		{
			wineLabel->x = 0; wineLabel->y = 0;
			wineLabel->height = image.size().height;
			wineLabel->width = image.size().width;
		}
	if(AnalyseImage(image,recognizedYears,wineLabel)>0)
	{
		//Returning max year
		unsigned int tmpYear = 1000;
		for(size_t i = 0; i<recognizedYears.size();i++)
		{
#ifdef _DEBUG_MODE_WIR_OCR
			cout<<"AI(M)"<<recognizedYears[i]<<endl;
#endif
			if(recognizedYears[i]>tmpYear)
				tmpYear = recognizedYears[i];
		}
		return tmpYear;
	}
	else
	{
#ifdef _DEBUG_MODE_WIR_OCR
		cout<<"Cannot detect image"<<endl;
		cout<<"Size "<<*wineLabel<<endl;
#endif
		if (wineLabel != NULL)
			if(wineLabel->x+wineLabel->width>image.size().width || wineLabel->y+wineLabel->height>image.size().height)
			{
				wineLabel->x = 0; wineLabel->y = 0;
				wineLabel->height = image.size().height;
				wineLabel->width = image.size().width;
			}
	}
	return -1;
};

int WIR_OCR::FindObjects(std::vector < std::vector<cv::Point2i > >& blobs, cv::Size imageSize, vector < cv::Rect > &contours )
{
	int allowedMargin;
	unsigned int correction = 1;

	contours.clear();
	for(size_t i=0; i <blobs.size() ; i++) {
		//if(blobs[i].size()<10)
		//	continue;
        //deleting really long objects
		int maxX = 0; int minX = imageSize.width;
		int maxY = 0; int minY = imageSize.height;
		for(size_t j=0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;
			if(x<minX) minX = x;
			if(x>maxX) maxX = x;
			if(y<minY) minY = y;
			if(y>maxY) maxY = y;
        }
		cv::Rect tmpRect;
		tmpRect.x = minX; tmpRect.y = minY;
		tmpRect.width = abs(maxX-minX)+correction;
		tmpRect.height = abs(maxY-minY)+correction;
		if(tmpRect.width>=minObjectWidth && tmpRect.height>=minObjectHeight)
		{
			// проверяем можем ли произвести слияния двух облостей;
			//По сути операцию нужно повторять несколько раз используя алгоритм Флойда Варшела, только это вычислительно сложно
			int isMerged = 0;
			for(size_t j=0; j<contours.size();j++)
			{
				allowedMargin = MAX(contours[j].height,tmpRect.height)/5;
				
				if(MIN(contours[j].x+contours[j].width,tmpRect.x+tmpRect.width)>MAX(contours[j].x, tmpRect.x)
					&&
					((contours[j].y<=tmpRect.y && contours[j].y+contours[j].height+allowedMargin>=tmpRect.y) ||
					(contours[j].y>=tmpRect.y && contours[j].y<=tmpRect.y+tmpRect.height+allowedMargin))
					)
				{
					//Merging
					isMerged = 1;
					int x = MIN(contours[j].x,tmpRect.x);
					int y = MIN(contours[j].y,tmpRect.y);
					int x2 = MAX(contours[j].x+contours[j].width, tmpRect.x+tmpRect.width);
					int y2 = MAX(contours[j].y+contours[j].height, tmpRect.y+tmpRect.height);
					contours[j].x = x;
					contours[j].y = y;
					contours[j].width = x2-x;
					contours[j].height = y2-y;
					break;
				}
			}
			if(!isMerged)
				contours.push_back(tmpRect);
		}
	}
	return contours.size();
}

void WIR_OCR::FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs)
{
    blobs.clear();

    // Fill the label_image with the blobs
    // 0  - background
    // 255  - unlabelled foreground
    // 1+ - labelled foreground

    cv::Mat label_image;
	binary.convertTo(label_image,CV_32FC1); // weird it doesn't support CV_32S!

    int label_count = 1; // starts at 2 because 0,1 are used already

    for(int y=0; y < binary.rows; y++) {
        for(int x=0; x < binary.cols; x++) {
            if(label_image.at<float>(y,x) != 255) {
				//cout<<label_image.at<float>(y,x)<< endl;
                continue;
            }

            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), cv::Scalar(label_count), &rect, cv::Scalar(0), cv::Scalar(0), 4);

            std::vector <cv::Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(label_image.at<float>(i,j) != label_count) {
                        continue;
                    }

                    blob.push_back(cv::Point2i(j,i));
                }
            }

            blobs.push_back(blob);

            label_count++;
        }
    }
}

int WIR_OCR::saveTrainingDB(const char* file_path)
{
	if(initialized)
		return WIR_OCR_NOT_INITIALIZED;
	if(InternalCurruptionCheck())
	{
		WIRInternalPanic();
		return -1;
	}
	FileStorage fs(file_path, FileStorage::WRITE);
	if(!fs.isOpened())
	{
	#ifdef _DEBUG_MODE_WIR_OCR
		cout<< "Cannot save settings to "<<file_path<<endl;
		WIRInternalPanic(WIRE_CANNOT_PROCESS_IO);
	#endif
		return -1;
	}
	fs<<"trainData"<<trainData;
	fs<<"trainClasses"<<trainClasses;
	fs.release();
	return 1;
};

int WIR_OCR::loadTrainingDB(const char* file_path)
{
	FileStorage fs(file_path, FileStorage::READ);
	if(!fs.isOpened())
	{
#ifdef _DEBUG_MODE_WIR_OCR
		cout<< "Cannot load settings from "<<file_path<<endl;
		WIRInternalPanic(WIRE_CANNOT_PROCESS_IO);
#endif
		return -1;
	}
#ifndef WIR_DO_NOT_DESTROY_KNEAREST
	delete knearest;
#endif
	fs["trainData"]>>trainData;
	fs["trainClasses"]>>trainClasses;
	knearest = new KNearest();
	if(knearest != NULL)
	{knearest->train(trainData, trainClasses); initialized = 1;}
	else
	{WIRInternalPanic(WIRE_NOT_ENOUGH_MEMORY); initialized = 0;}
	fs.release();
	return initialized;
};
void WIR_OCR::WIRInternalPanic(int type)
{
	#ifdef _DEBUG_MODE_WIR_OCR
		std::cout<<"Something has gone wrong!!! ERROR CODE:"<<type<<endl;
	#endif
	if(type<0)
	{
#ifndef WIR_DO_NOT_DESTROY_KNEAREST
		delete knearest;
#endif		
	}
	if (errorCallback != NULL)
		errorCallback(type);
}

void WIR_OCR::RegionColorize(cv::Size inputSize, cv::Mat& output,  std::vector < std::vector<cv::Point2i > >& blobs,
		int& maxX, int& minX, int& maxY, int& minY)
{
	maxX = 0; maxY = 0;
	minX = inputSize.width; minY = inputSize.height;
	for(size_t i=0; i < blobs.size(); i++) {
		if(blobs[i].size()<10)
			continue;
		for(size_t j=0; j < blobs[i].size(); j++) {
			int x = blobs[i][j].x;
			int y = blobs[i][j].y;
			output.at<unsigned char>(y,x) = 255;
			if (maxX < x) maxX = x;
			if (maxY < y) maxY = y;
			if (minX > x) minX = x;
			if (minY > y) minY = y;
		}
	}
}

void WIR_OCR::RegionColorize(cv::Mat& output,  std::vector < std::vector<cv::Point2i > >& blobs)
{
	for(size_t i=0; i < blobs.size(); i++) {
		if(blobs[i].size()<10)
			continue;
		for(size_t j=0; j < blobs[i].size(); j++) {
			int x = blobs[i][j].x;
			int y = blobs[i][j].y;
			output.at<unsigned char>(y,x) = 255;
		}
	}
}
void WIR_OCR::RegionColorize(cv::Mat& output,  std::vector<cv::Point2i >& blob)
{
		for(size_t j=0; j < blob.size(); j++) {
			int x = blob[j].x;
			int y = blob[j].y;
			output.at<unsigned char>(y,x) = 255;
		}
}

//Returning index;
int WIR_OCR::FindMaxBlob(const std::vector < std::vector<cv::Point2i > >& blobs)
{
	int maxSizeIndex = 0;
	if(blobs.size() > 0)
	{
		for(size_t i = 0; i<blobs.size(); i++)
			if(blobs[i].size()>blobs[maxSizeIndex].size())
				maxSizeIndex = i;
		return maxSizeIndex;
	}
	else
		return -1;
}

cv::Rect WIR_OCR::GetBlobRect(const std::vector<cv::Point2i >& blob, bool centered)
{
	const double MAX_DIVIATION = 0.1;
	const double DIVIATION_MARGIN = 0.02;
	
	int maxX = INT_MIN; int maxY = INT_MIN;
	int minX = INT_MAX; int minY = INT_MAX;
	long long xCounter = 0;
	if(blob.size() == 0)
	{
		maxX = 0; minX = 0; maxY = 0; maxX = 0;
	};

	for(size_t i = 0; i< blob.size(); i++)
	{
		int x = blob[i].x;
		xCounter+=x;
		int y = blob[i].y;
		if (maxX < x) maxX = x;
		if (maxY < y) maxY = y;
		if (minX > x) minX = x;
		if (minY > y) minY = y;
	}
	xCounter/=(long long)blob.size();
	if (centered)
	{
		int deltaR = maxX - (int) xCounter;
		int deltaL = (int) xCounter - minX;
		if((maxX - minX)*MAX_DIVIATION < abs(deltaR - deltaL))
		{
			int diviationMargin = (maxX - minX)*DIVIATION_MARGIN;
			maxX = MIN((int)xCounter + deltaR, (int) xCounter + MIN(deltaR,deltaL) + diviationMargin);
			minX = MAX((int)xCounter - deltaL, (int)xCounter - MIN(deltaL, deltaR) - diviationMargin);
		};
	};

	return cv::Rect(minX, minY, maxX-minX, maxY-minY);
}
