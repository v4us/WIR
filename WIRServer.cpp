#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <signal.h>
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

static WIR01 *classifier;

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
        classifier->Recognize(request_info->query_string,results,MAX_NUMBER_OF_RESULTS);
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
  }
  if (strcmp(request_info->uri,"/ping") == 0)
  {
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


int main(void) {
        
      
    WIR01 classifier2;    
    if ( classifier2.loadTrainingDB("/home/ubuntu/winee/WIR01/data/test_data.xml")<0)
    {  cout<<"Cannot load file"<<endl;  return -1;}
   classifier = &classifier2;    
        struct mg_context *ctx;
  struct mg_callbacks callbacks;

  // List of options. Last element must be NULL.
  const char *options[] = {"listening_ports", "8080", NULL};
  // Prepare callbacks structure. We have only one callback, the rest are NULL.
  memset(&callbacks, 0, sizeof(callbacks));
  callbacks.begin_request = begin_request_handler;

  // Start the web server.
  ctx = mg_start(&callbacks, NULL, options);
        /* The Big Loop */
  getchar();
  fflush(stdout);
  mg_stop(ctx);
  printf("%s", " done.\n");

  return EXIT_SUCCESS;

}
