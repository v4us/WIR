#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "WIR01.h"
#include "WIR_OCR.h"
#include <dirent.h>
#define _CRT_SECURE_NO_WARNINGS
#define _ATL_SECURE_NO_WARNINGS

using namespace std;
using namespace cv;

void readme();

/*void memory_info(){

    int tSize = 0, resident = 0, share = 0;
    ifstream buffer("/proc/self/statm",ios_base::in);
    buffer >> tSize >> resident >> share;
    buffer.close();

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    double rss = resident * page_size_kb;
    cout << "RSS - " << rss << " kB\n";

    double shared_mem = share * page_size_kb;
    cout << "Shared Memory - " << shared_mem << " kB\n";

    cout << "Private Memory - " << rss - shared_mem << "kB\n";
}*/
/**
 * @function main
 * @brief Main function
 */
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
  WIRTrainSample tmpTrainSample;
  vector<WIRTrainSample> trainSamples;
  dirSpec[0]=0;
  strcpy(dirSpec,argv[2]);
  DIR *dir;
  struct dirent *ent;
  //��������� ���� �� ���� ������ � �����
  if ((dir = opendir (dirSpec)) != NULL) {
	  // print all the files and directories within directory
	  while ((ent = readdir (dir)) != NULL) {
		  if (strlen(ent->d_name)<4)
			  continue;
		  if (ent->d_name[0] == '.')
			  continue;
		  if (ent->d_type == DT_REG)
		  {
				tmpTrainSample.classLabel = 0; //not used here;
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

				//cout << tmpTrainSample.imagePath << endl;
				trainSamples.push_back(tmpTrainSample);
				//std::cout << trainSamples.size() << std::endl;
		  }
	  }
	  closedir (dir);

	} 
  else 
  {
	 cout << "No pictures have been found" <<endl;
	 readme();
	return -1;
	}

	vector<WIRResult> results;
	 //��� ���������� �������������� � ������������ ������������� ������������ 
	//��������� ����������� ���������� �������������� ������� � ������ �� ��������.
	//������ ���� ������ ���� ���������� �� FALSE ��� ������ �� �������
	//classificator.setHistogramUse(true);
	//��������� �� ��������� ������
	classificator.addTrainSamples(trainSamples);
	//Deviding the learning sequence into parts
	/*std::cout<<"Preparing Learning Data : "<<trainSamples.size()<<std::endl;
	vector<WIRTrainSample> tmpTrainSamples;
	for (unsigned int i = 0;i<trainSamples.size(); i++)
	{
		  //std::cout<<i<<std::endl;
		  tmpTrainSamples.push_back(trainSamples[i]);
		  if (tmpTrainSamples.size() == 100)
		  {
		  	classificator.addTrainSamples(tmpTrainSamples);
		  	std::cout << (double)i /trainSamples.size() <<std::endl;
		  	sleep(1);
			tmpTrainSamples.clear();
			//memory_info();
		  }
	}
	classificator.addTrainSamples(tmpTrainSamples);
	*/
	//�������� ������ � �������� ������� �� ����������. �������� ��������������: ��������� ����������.
	//classificator.LoadBinary("C:\\LGP500\\1");

	//������� ���������� ������� ����� ���������� ��� ��������� ������ ���������������
	//classificator.loadTrainingDB("/home/ubuntu/winee/WIR01/data/test_data.xml");
	//classificator.loadTrainingDB("./test_data.xml");
	cout<<"LOADED"<<endl;
	if(classificator.Recognize(argv[1],results,3))
	{
		Mat img_object = imread(argv[1], IMREAD_GRAYSCALE );
		Mat img_match = imread(results[0].filePath);
		cout << "File "<< argv[1] <<" Matchs "<< results[0].fileName << endl;
		cout <<"Detected Year: "<< results[0].year<<endl;
		cout <<"Hist: "<< results[0].hist<<endl;
		if(!img_object.empty() && !img_match.empty())
		{
			//imshow("Match",img_match);
			//imshow("Object", img_object);
		}
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
		img_match = imread(results[minId].filePath);
		cout << "File "<< argv[1] <<" Matchs "<< results[minId].fileName << endl;
		cout <<"Detected Year: "<< results[minId].year<<endl;
		cout <<"Hist: "<< results[minId].hist<<endl;
		//if(!img_match.empty())
		//	imshow("Match HIst",img_match);
	}
	else
		cout<<"No match has been found"<<endl;
	//��������� ��������� ��������������
	//classificator.saveTrainingDB("/home/ubuntu/winee/WIR01/test_data.xml");
	classificator.SaveBinary("/home/ubuntu/winee/WIR01/saved");
	cout<<"SAVED"<<endl;
	//���������� ���������� �� ������ ���������� ������. 
	//vector<const char*> inputNames; inputNames.push_back("C:\\LGP500");
	//WIR01::GenerateUpdates(param,trainSamples,inputNames,10,10,"Test");

	//��������� ������ � �������� ������� ������������ � �������� ��� ��������� ���������.
	//classificator.SaveBinary("C:\\LGP500\\1");
	//cv::waitKey(0);
	/*double hitRate,firstHitRate,firstClassHitRate, classMatchHitRate;
	classificator.RecognitionTest(hitRate, firstHitRate, firstClassHitRate, classMatchHitRate);
	cout << "HitRate : "<<hitRate<<endl;
	cout << "firstHitRate : "<<firstHitRate<<endl;
	cout << "firstClassHitRate : "<<firstClassHitRate << endl;
	cout << "cvlassMatchHitRate : " << classMatchHitRate << endl;
*/
  return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SIF01 <image> <lib_path>" << std::endl; }




