#pragma once

// #define __DEBUG__
#define __DEBUG__D false
#define MAX_DATA INT_MAX
#define eps 1e-6

#define MIN(a, b) (((a) < (b) ? (a) : (b)))
#define MAX(a, b) (((a) > (b) ? (a) : (b)))

typedef unsigned long long int uint64;
typedef unsigned int uint;

enum LogPriorityEnum
{
  critical,
  warn,
  error,
  info,
  debug,
  none
};
