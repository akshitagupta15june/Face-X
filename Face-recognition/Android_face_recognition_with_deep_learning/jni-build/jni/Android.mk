LOCAL_PATH:=$(call my-dir)

#Module jnilibsvm

include $(CLEAR_VARS)

LOCAL_MODULE	:= jnilibsvm
LOCAL_CFLAGS    := -DDEV_NDK=1
LOCAL_SRC_FILES := \
	common.cpp jnilibsvm.cpp \
	libsvm/svm-train.cpp \
	libsvm/svm-predict.cpp \
	libsvm/svm.cpp

LOCAL_LDLIBS	+= -llog -ldl

include $(BUILD_SHARED_LIBRARY)