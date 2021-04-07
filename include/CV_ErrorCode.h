/******************************************************************
 * 定义错误码
 * created: zhangwenguang
 * date:11/08/2019
 * version:0.0.1
 *****************************************************************/
#ifndef _CVERRORCODE_
#define _CVERRORCODE_

#include <stdio.h>

/********************返回值列表***********************/
/*处理成功*/
#define CODE_CV_OK                              0x00
/*处理错误*/
#define CODE_CV_ERROR                      0x01
/*内存溢出异常*/
#define CODE_CV_OUT_OF_MEM       0x02
/*无效参数异常*/
#define CODE_CV_INVALID_PARAM   0x03
/*空指针异常*/
#define CODE_CV_NULL_POINTER    0x04
/****************************************************/

/*自定义断言，返回错误码*/
#define CODECVASSERT(x,y)                 if (!(x)) {printf("CODE_CV_ERROR: erroe code is 0x%x\n", y); return y; }



#endif  // _CVERRORCODE_
