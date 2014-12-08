#ifndef _SIMPLE_TIMER
#define _SIMPLE_TIMER

#include <Windows.h>

class SimpleTimer
{
public:
	SimpleTimer( void );
	~SimpleTimer( void );

	void start( void );
	float stop( void );

private:
	__int64 GetTimeMs64( void );

	__int64 start_time;
};

#endif