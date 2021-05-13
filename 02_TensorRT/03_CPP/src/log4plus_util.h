#ifndef _LOG4PLUS_UTIL_H_
#define _LOG4PLUS_UTIL_H_

#include <string>

#define DEBUG(input) DEBUG_LOG((input), (__FILE__), (__LINE__));
#define ERROR(input) ERROR_LOG((input), (__FILE__), (__LINE__));
#define WARN(input) WARN_LOG((input), (__FILE__), (__LINE__));

using namespace std;

void DEBUG_LOG(const std::string& strInfo, const std::string& strFile, int iLine);
void ERROR_LOG(const std::string& strInfo, const std::string& strFile, int iLine);
void WARN_LOG(const std::string& strInfo, const std::string& strFile, int iLine);

class CLog
{
public:
    static CLog* Initialize(const std::string& strLogFile);

private:
    CLog(const std::string& strLogFile);
    static CLog* pInstance;
};

#endif
