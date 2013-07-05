#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <syslog.h>
#include <string.h>
#include <sstream>
#include "mongoose.h"
#include "WIR01.h"

#define MAX_NUMBER_OF_RESULTS 5
using namespace std;

template <class T>
inline std::string toString (const T& t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}

WIR01 classifier;
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
        vector<WIRResult> results;
        classifier.Recognize(request_info->query_string,results,MAX_NUMBER_OF_RESULTS);
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
      if (request_info->query_string == NULL)
      {
          noFoundReplay(conn);
          return 1;
      };

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
  noFoundReplay(conn);
  return 1;
}


int main(void) {
        
        /* Our process ID and Session ID */
        pid_t pid, sid;
        
        std::cout<<"Preparing to load data"<<std::endl;
        if ( classifier.loadTrainingDB("/home/ubuntu/winee/WIR01/data/test_data.xml")<0)
        exit(EXIT_FAILURE);
        std::cout<<"Loaded"<<std::endl;
        /* Fork off the parent process */
        pid = fork();
        if (pid < 0) {
                exit(EXIT_FAILURE);
        }
        /* If we got a good PID, then
           we can exit the parent process. */
        if (pid > 0) {
                exit(EXIT_SUCCESS);
        }

        /* Change the file mode mask */
        umask(0);
                
        /* Open any logs here */        
                
        /* Create a new SID for the child process */
        sid = setsid();
        if (sid < 0) {
                /* Log the failure */
                exit(EXIT_FAILURE);
        }
        

        std::couut<<"initializaed"<<std::endl;
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

  // List of options. Last element must be NULL.
  const char *options[] = {"listening_ports", "8080", NULL};
  // Prepare callbacks structure. We have only one callback, the rest are NULL.
  memset(&callbacks, 0, sizeof(callbacks));
  callbacks.begin_request = begin_request_handler;

  // Start the web server.
  ctx = mg_start(&callbacks, NULL, options);
  if (ctx == NULL)
    return EXIT_FAILURE;
        /* The Big Loop */
        while (exit_flag) {
           /* Do some task here ... */
           
           sleep(1); /* wait 30 seconds */
        }

        fflush(stdout);
  mg_stop(ctx);
  printf("%s", " done.\n");
   exit(EXIT_SUCCESS);
}
