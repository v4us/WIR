#include <sys/types.h>
#include <time.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <syslog.h>
#include <string.h>
#include <sstream>
#include <fstream>
#include "mongoose.h"
#include "WIR01.h"

#define MAX_NUMBER_OF_RESULTS 5
using namespace std;
char* defTime = "00:00:00";

//Returning current time as a string
inline char* GetCurrentTimeA(void)
{
   time_t current_time;
   char* c_time_string;
 
    /* Obtain current time as seconds elapsed since the Epoch. */
    current_time = time(NULL);
    if (current_time == ((time_t)-1))
      return defTime;
    /* Convert to local time format. */
    c_time_string = ctime(&current_time);
    if (c_time_string == NULL)
      return defTime;
    else
    {
      int c_len = strlen(c_time_string);
      if (c_len >=2)
        c_time_string[c_len - 2] = 0;
      return c_time_string;
    }
}

template <class T>
inline std::string toString (const T& t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}

static WIR01* classifier;
static int exit_flag = 1;

static void noFoundReplay (struct mg_connection *conn)
{
    mg_printf(conn, "HTTP/1.1 404 Not Found\n");
}
// This function will be called by mongoose on every new request.
static int begin_request_handler(struct mg_connection *conn) {
  const struct mg_request_info *request_info = mg_get_request_info(conn);
  if (strcmp(request_info->uri,"/image-process") == 0)
  {
      if (request_info->query_string == NULL)
      {
          noFoundReplay(conn);
          return 1;
      };
        std::cout<<"["<<GetCurrentTimeA()<<"] ";
        std::cout<<"Request recived from : " << request_info->remote_ip << std::endl;
        std::cout<<"Requested Data : " << request_info->query_string << std::endl;
        vector<WIRResult> results;
        classifier->Recognize(request_info->query_string,results,MAX_NUMBER_OF_RESULTS);
        
        std::cout<<"["<<GetCurrentTimeA()<<"] ";
        std::cout<<"Request from " << request_info->remote_ip << " has been processed" <<std::endl;
        std::cout<<"Detected : " <<results.size() << std::endl;
        // Prepare the message we're going to send
        std::ostringstream ss;
        WIRResult::vectorOutput(ss,results);

        // Send HTTP reply to the client
        mg_printf(conn,
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: %d\r\n"        // Always set Content-Length
                "\r\n"
                "%s",
                (int)ss.str().length(), ss.str().c_str());
  // Returning non-zero tells mongoose that our function has replied to
  // the client, and mongoose should not send client any more data.
    return 1;
  };
  if (strcmp(request_info->uri,"/exit") == 0)
  {
      if (request_info->query_string == NULL)
      {
          noFoundReplay(conn);
          return 1;
      };
        std::cout<<"["<<GetCurrentTimeA()<<"] ";
        std::cout<<"TERMINATED BY USER REQUEST"<<std::cout;
        std::cout<<"Exit CODE : " << request_info ->query_string <<std::cout;
	// Send HTTP reply to the client
        mg_printf(conn,
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: %d\r\n"        // Always set Content-Length
                "\r\n"
                "%s",
                strlen("\"Status\" : \"OK\""), "\"Status\" : \"OK\"");
        exit_flag = 0;
  // Returning non-zero tells mongoose that our function has replied to
  // the client, and mongoose should not send client any more data.
    return 1;
  };
  if (strcmp(request_info->uri,"/ping") == 0)
  {
        std::cout<<"["<<GetCurrentTimeA()<<"] PING"<<std::endl;
        // Send HTTP reply to the client
        mg_printf(conn,
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: %d\r\n"        // Always set Content-Length
                "\r\n"
                "%s",
                strlen("\"Status\" : \"OK\""), "\"Status\" : \"OK\"");
  // Returning non-zero tells mongoose that our function has replied to
  // the client, and mongoose should not send client any more data.
    return 1;
  };
  noFoundReplay(conn);
  return 1;
}


int main( int argc, char** argv )
{
        
        char dbPath[1024];
        dbPath[0]=0;
        if( argc == 2 )
        {
          strcpy(dbPath, argv[1]);
        }
        else
        {
          strcpy(dbPath, "/home/ubuntu/winee/WIR01/saved_rono5");
        }
        /* Our process ID and Session ID */
        pid_t pid, sid;
        pid = 0; sid = 0;
        
        /* Fork off the parent process */
        pid = fork();
        if (pid < 0) {
		printf("YOU ARE FUCKED!");
                exit(EXIT_FAILURE);
        }
        /* If we got a good PID, then
           we can exit the parent process. */
        if (pid > 0) {
		printf("process_id of child process %d \n", pid);
                exit(EXIT_SUCCESS);
        }

        /* Change the file mode mask */
        umask(0);
                
        /* Open any logs here */        
        std::ofstream outFileS;
        outFileS.open("/home/ubuntu/winee/WIR01/DaemonLog.txt");
        outFileS<<"LOG Started"<<std::endl;
        outFileS.flush();
        std::streambuf *localSB =  outFileS.rdbuf();
        std::cout.rdbuf(localSB);

        /* Create a new SID for the child process */
        sid = setsid();
        if (sid < 0) {
                /* Log the failure */
                exit(EXIT_FAILURE);
        }
        

        std::cout<<"initializaed"<<std::endl;
        /* Change the current working directory */
        if ((chdir("/")) < 0) {
                /* Log the failure */
                exit(EXIT_FAILURE);
        }
        
        /* Close out the standard file descriptors */
        close(STDIN_FILENO);
        //close(STDOUT_FILENO);
        //close(STDERR_FILENO);
        
        /* Daemon-specific initialization goes here */
        struct mg_context *ctx;
        struct mg_callbacks callbacks;
        WIR01 classifier2;
        classifier = &classifier2;
  classifier2.SetUseClustering(true);
  classifier2.SetPreCropping(false);
  classifier2.SetCropping(false);
  classifier2.SetAfterCCropping(true); //true
  classifier2.SetPushSameClassImages(false); //true
  std::cout<<"Preparing to load data..."<<std::endl;
  //if (classifier2.loadTrainingDB("/home/ubuntu/winee/WIR01/test_data.xml")<0)
  if(classifier2.LoadBinary(dbPath)<0)
    exit(EXIT_FAILURE);
  std::cout<<"Loaded : " <<dbPath<<std::endl;

  // List of options. Last element must be NULL.
  const char *options[] = {"listening_ports", "8080", NULL};
  // Prepare callbacks structure. We have only one callback, the rest are NULL.
  memset(&callbacks, 0, sizeof(callbacks));
  callbacks.begin_request = begin_request_handler;

  // Start the web server.
  ctx = mg_start(&callbacks, NULL, options);
  if (ctx == NULL)
    exit(EXIT_FAILURE);
        /* The Big Loop */
        while (exit_flag) {
           /* Do some task here ... */
           //outFileS << "Test"<<std::endl;
           sleep(1); /* wait 30 seconds */
        }
  outFileS.close();
  mg_stop(ctx);
   exit(EXIT_SUCCESS);
}
