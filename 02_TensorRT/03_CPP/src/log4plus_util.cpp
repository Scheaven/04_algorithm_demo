#include "log4plus_util.h"
#include <sstream>
#include <memory>
#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>
#include <log4cplus/fileappender.h>
#include <log4cplus/configurator.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <iostream>
using namespace std;
using namespace log4cplus;

static Logger global_pLogger;

CLog* CLog::pInstance = NULL;

CLog* CLog::Initialize(const string& strLogFile)
{
    if (NULL == pInstance)
    {
        pInstance = new CLog(strLogFile);
    }
    return pInstance;
}

CLog::CLog(const string& strLogFile)
{
    PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT(strLogFile));
    global_pLogger = Logger::getRoot();
}

void DEBUG_LOG(const string& strInfo, const string& strFile, int iLine)
{
    stringstream ssLogData;
    ssLogData << "[" << strFile << ":" << iLine << "] " << strInfo;
    std::cout << ssLogData.str() << std::endl;
    LOG4CPLUS_DEBUG(global_pLogger, ssLogData.str());
}

void ERROR_LOG(const string& strInfo, const string& strFile, int iLine)
{
    stringstream ssLogData;
    ssLogData << "[" << strFile << ":" << iLine << "] " << strInfo;
    std::cout << ssLogData.str() << std::endl;
    LOG4CPLUS_ERROR(global_pLogger, ssLogData.str());
}

void WARN_LOG(const string& strInfo, const string& strFile, int iLine)
{
    stringstream ssLogData;
    ssLogData << "[" << strFile << ":" << iLine << "] " << strInfo;
    std::cout << ssLogData.str() << std::endl;
    LOG4CPLUS_WARN(global_pLogger, ssLogData.str());
}

