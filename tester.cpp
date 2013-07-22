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
char tmpID[128];
  DIR *dir;
  struct dirent *ent;
  WIRTrainSample tmpTrainSample;
  //получаеми путь ко всем файлам в папке
  cout<<"Opening "<<argv[2]<<endl;
  if ((dir = opendir (argv[2])) != NULL) {
  	cout <<"Opened"<<endl;
	  // print all the files and directories within directory
	  while ((ent = readdir (dir)) != NULL) {
		  if (ent->d_name[0] == '.')
			  continue;
		  if (strlen(ent->d_name)<4) 
			  continue;
		  if (ent->d_type == DT_REG)
		  {
				tmpID[0]=0;
				strncpy(tmpID, ent->d_name, strlen(ent->d_name)-3);
				tmpTrainSample.classLabel = atoi(tmpID); //not used here;
				tmpTrainSample.imagePath[0] = 0;
				tmpTrainSample.imageName[0] = 0;
				strcpy(tmpTrainSample.imagePath, argv[2]);
#ifdef __WIN__
                strcat(tmpTrainSample.imagePath,"\\");
#endif
#ifdef __LINUX__
                strcat(tmpTrainSample.imagePath, "/");
#endif

				strcat(tmpTrainSample.imagePath,ent->d_name);
				strcpy(tmpTrainSample.imageName,ent->d_name);
				
				cout << tmpTrainSample.imagePath << endl;
				trainSamples.push_back(tmpTrainSample);
		  }
	  }
	  closedir (dir);

	}
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
	classificator.LoadBinary(argv[1]);
	time_t timer;
	timer = time(NULL);
	cout<<"LOADED"<<endl;
	double hitRate,firstHitRate,firstClassHitRate, classMatchHitRate;
	classificator.RecognitionTest(trainSamples, hitRate, firstHitRate, firstClassHitRate, classMatchHitRate);
	cout << "HitRate : "<<hitRate<<endl;
	cout << "firstHitRate : "<<firstHitRate<<endl;
	cout << "firstClassHitRate : "<<firstClassHitRate << endl;
	cout << "cvlassMatchHitRate : " << classMatchHitRate << endl;
	return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SIF01 <image> <lib_path>" << std::endl; }




