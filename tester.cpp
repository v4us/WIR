#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <time.h>
#include "WIR01.h"
#include "WIR_OCR.h"
#include <dirent.h>
#define _CRT_SECURE_NO_WARNINGS
#define _ATL_SECURE_NO_WARNINGS

using namespace std;
using namespace cv;

void readme();

int main( int argc, char** argv )
{
  if( argc != 3 )
  { readme(); return -1; }
  //WIR_OCR tmpOCR;
  //tmpOCR.m();
  //tmpOCR.saveTrainingDB("OCR.XML");
  //additional data
  cout<<"Please select mode:\n 1 standart\n2 fast\n3 DFast"<<endl;
  int workingMode;
  cin>>workingMode;

  WIRParam param;
  switch (workingMode)
  {

  case 1:
	  param = WIRParam(standartWIR);
	  break;
  case 2:
	  param = WIRParam(fastWIR);
	  break;
  case 3:
	  param = WIRParam(doubleFastWIR);
	  break;
  default:
	  param = WIRParam();
  };
  param.labelExtraction = WIR_EL_SOFT;
  char dirSpec[2048];
  WIR01 classificator(param);
//  WIRTrainSample tmpTrainSample;
  vector<WIRTrainSample> trainSamples;
  dirSpec[0]=0;

	vector<WIRResult> results;
	//classificator.SetUseClustering(true);
	 //äëÿ ïðîâåäåíèÿ ýêñìïåðåìåíòîâ è èññëåäîâàíèÿ óñòàíàâëèâàåò îïöèîíàëüíîå 
	//ñðàâíåíèå ãèñòîãðàììû ôðàãìåíòîâ ïðåäñòàâëÿþùèõ èíòåðåñ â êàæäîì èç ìîìåíòîâ.
	//ñåé÷àñ ôëàã äîëæåí áûòü óñòàíîâëåí íà FALSE ïðè ðàáîòå íà ñåðâåðå
	//classificator.setHistogramUse(true);
	classificator.SetUseClustering(true);
	classificator.SetPreCropping(false);
	classificator.SetCropping(false);
	//îáó÷àåìñÿ íà ñîçäàííûõ ôàéëàõ
	//classificator.addTrainSamples(trainSamples);
	classificator.LoadBinary("/home/ubuntu/winee/WIR01/saved_rono");
	time_t timer;
	timer = time(NULL);
	cout<<"LOADED"<<endl;
	if(classificator.Recognize(argv[1],results,3))
	{
		//Mat img_object = imread(argv[1], IMREAD_GRAYSCALE );
		//Mat img_match = imread(results[0].filePath);
		cout << "File "<< argv[1] <<" Matchs "<< results[0].fileName << endl;
		cout <<"Detected Year: "<< results[0].year<<endl;
		cout <<"Hist: "<< results[0].hist<<endl;
		/*if(!img_object.empty() && !img_match.empty())
		{
			//imshow("Match",img_match);
			//imshow("Object", img_object);
		}*/
		//Histogramm
		cout<<"Histograms data"<<endl;
		unsigned int minId = 0;
		for(size_t i = 0; i<results.size(); i++)
		{
			if(results[i].hist<results[minId].hist)
				minId = i;
			cout<<"#"<<i<<" : " <<results[i].hist<<endl;
		}
		cout << "Best Hist feeting"<<endl;
		cout << "-------------------------------------------------------"<<endl;
		//img_match = imread(results[minId].filePath);
		cout << "File "<< argv[1] <<" Matchs "<< results[minId].fileName << endl;
		cout <<"Detected Year: "<< results[minId].year<<endl;
		cout <<"Hist: "<< results[minId].hist<<endl;

	}
	else
		cout<<"No match has been found"<<endl;
  return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SIF01 <image> <lib_path>" << std::endl; }




