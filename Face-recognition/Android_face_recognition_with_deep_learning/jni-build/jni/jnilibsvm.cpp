/* Copyright 2016 Michael Sladoje and Mike Schälchli. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <jni.h>
#include <string.h>
#include <android/log.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "./libsvm/svm-train.h"
#include "./libsvm/svm-predict.h"
#include "common.h"

/***************************************************************************************
 *    Title: AndroidLibSvm
 *    Author: yctung
 *    Date: 16.09.2015
 *    Code version: -
 *    Availability: https://github.com/
 *
 ***************************************************************************************/

// helper function to be called in Java for making svm-train
extern "C" void Java_ch_zhaw_facerecognitionlibrary_Recognition_SupportVectorMachine_jniSvmTrain(JNIEnv *env, jobject obj, jstring cmdIn){
	const char *cmd = env->GetStringUTFChars(cmdIn, 0);
	debug("jniSvmTrain cmd = %s", cmd);

	std::vector<char*> v;

	// add dummy head to meet argv/command format
	std::string cmdString = std::string("dummy ")+std::string(cmd);

	cmdToArgv(cmdString, v);

	// make svm train by libsvm
	svmtrain::main(v.size(),&v[0]);


	// free vector memory
	for(int i=0;i<v.size();i++){
		free(v[i]);
	}

	// free java object memory
	env->ReleaseStringUTFChars(cmdIn, cmd);
}

// helper function to be called in Java for making svm-predict
extern "C" void Java_ch_zhaw_facerecognitionlibrary_Recognition_SupportVectorMachine_jniSvmPredict(JNIEnv *env, jobject obj, jstring cmdIn){
	const char *cmd = env->GetStringUTFChars(cmdIn, 0);
	debug("jniSvmPredict cmd = %s", cmd);

	std::vector<char*> v;

	// add dummy head to meet argv/command format
	std::string cmdString = std::string("dummy ")+std::string(cmd);

	cmdToArgv(cmdString, v);

	// make svm train by libsvm
	svmpredict::main(v.size(),&v[0]);


	// free vector memory
	for(int i=0;i<v.size();i++){
		free(v[i]);
	}

	// free java object memory
	env->ReleaseStringUTFChars(cmdIn, cmd);
}



/*
*  just some test functions -> can be removed
*/
extern "C" JNIEXPORT int JNICALL Java_ch_zhaw_facerecognitionlibrary_Recognition_SupportVectorMachine_testInt(JNIEnv * env, jobject obj){
	return 5566;
}

extern "C" void Java_ch_zhaw_facerecognitionlibrary_Recognition_SupportVectorMachine_testLog(JNIEnv *env, jobject obj, jstring logThis){
	const char * szLogThis = env->GetStringUTFChars(logThis, 0);
	debug("%s",szLogThis);

	env->ReleaseStringUTFChars(logThis, szLogThis);
}
