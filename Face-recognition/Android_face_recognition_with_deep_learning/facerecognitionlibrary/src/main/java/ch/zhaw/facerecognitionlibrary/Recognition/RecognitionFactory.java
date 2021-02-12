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

package ch.zhaw.facerecognitionlibrary.Recognition;

import android.content.Context;
import android.content.res.Resources;

import ch.zhaw.facerecognitionlibrary.R;

public class RecognitionFactory {
    public static Recognition getRecognitionAlgorithm(Context context, int method, String algorithm) {
        Resources resources = context.getResources();
        if (algorithm.equals(resources.getString(R.string.eigenfaces))){
            return new Eigenfaces(context, method);
        } else if (algorithm.equals(resources.getString(R.string.imageReshaping))){
            return new SupportVectorMachine(context, method);
        }
        else if (algorithm.equals(resources.getString(R.string.tensorflow))){
            return new TensorFlow(context, method);
        } else if (algorithm.equals(resources.getString(R.string.caffe))) {
            return new Caffe(context, method);
        } else {
            return null;
        }
    }
}
