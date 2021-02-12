package ch.zhaw.facerecognitionlibrary.Helpers;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.Resources;
import android.preference.PreferenceManager;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import ch.zhaw.facerecognitionlibrary.R;

/**
 * Created by sladomic on 05.11.16.
 */

public class PreferencesHelper {
    public enum Usage {RECOGNITION, DETECTION};
    SharedPreferences sharedPreferences;
    Resources resources;
    
    public PreferencesHelper(Context context){
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
        resources = context.getResources();
    }
    

    public String getClassificationMethod(){
        return sharedPreferences.getString("key_classification_method", resources.getString(R.string.eigenfaces));
    }

    public boolean getClassificationMethodTFCaffe(){
        return sharedPreferences.getBoolean("key_classificationMethodTFCaffe", true);
    }

    public float getGamma(){
        return Float.valueOf(sharedPreferences.getString("key_gamma", resources.getString(R.string.gamma)));
    }

    public  double[] getSigmas(){
        String[] sigmasString = sharedPreferences.getString("key_sigmas", resources.getString(R.string.sigmas)).split(",");
        if(sigmasString.length != 2){
            sigmasString = resources.getString(R.string.sigmas).split(",");
        }
        double[] sigmas = new double[3];
        for(int i=0; i<2; i++){
            sigmas[i] = Double.parseDouble(sigmasString[i]);
        }
        return sigmas;
    }

    public boolean getEyeDetectionEnabled(){
        return sharedPreferences.getBoolean("key_eye_detection", true);
    }

    public List<String> getStandardPreprocessing(Usage usage){
        if (usage == Usage.RECOGNITION){
            return getPreferenceList("key_standard_pre");
        } else if (usage == Usage.DETECTION){
            return getPreferenceList("key_detection_standard_pre");
        } else {
            return new ArrayList<>();
        }
    }

    public List<String> getBrightnessPreprocessing(Usage usage){
        if (usage == Usage.RECOGNITION){
            return getPreferenceList("key_brightness");
        } else if (usage == Usage.DETECTION){
            return getPreferenceList("key_detection_brightness");
        } else {
            return new ArrayList<>();
        }
    }

    public List<String> getContoursPreprocessing(Usage usage){
        if (usage == Usage.RECOGNITION){
            return getPreferenceList("key_contours");
        } else if (usage == Usage.DETECTION){
            return getPreferenceList("key_detection_contours");
        } else {
            return new ArrayList<>();
        }
    }

    public List<String> getContrastPreprocessing(Usage usage){
        if (usage == Usage.RECOGNITION){
            return getPreferenceList("key_contrast");
        } else if (usage == Usage.DETECTION){
            return getPreferenceList("key_detection_contrast");
        } else {
            return new ArrayList<>();
        }
    }

    public List<String> getStandardPostprocessing(Usage usage){
        if (usage == Usage.RECOGNITION){
            return getPreferenceList("key_standard_post");
        } else if (usage == Usage.DETECTION){
            return getPreferenceList("key_detection_standard_post");
        } else {
            return new ArrayList<>();
        }
    }

    private List<String> getPreferenceList(String key){
        Set<String> set = sharedPreferences.getStringSet(key, null);
        ArrayList<String> list;
        if(set != null) {
            list = new ArrayList<String>(set);
            Collections.sort(list);
            return list;
        } else {
            return new ArrayList<>();
        }
    }

    public String getCaffeModelFile(){
        return sharedPreferences.getString("key_modelFileCaffe", resources.getString(R.string.modelFileCaffe));
    }

    public String getCaffeWeightsFile(){
        return sharedPreferences.getString("key_weightsFileCaffe", resources.getString(R.string.weightsFileCaffe));
    }

    public String getCaffeOutputLayer(){
        return sharedPreferences.getString("key_outputLayerCaffe", resources.getString(R.string.weightsFileCaffe));
    }

    public float[] getCaffeMeanValues(){
        String[] meanValuesString = sharedPreferences.getString("key_meanValuesCaffe", resources.getString(R.string.meanValuesCaffe)).split(",");
        if(meanValuesString.length != 3){
            meanValuesString = resources.getString(R.string.meanValuesCaffe).split(",");
        }
        float[] meanValues = new float[3];
        for(int i=0; i<3; i++){
            meanValues[i] = Float.parseFloat(meanValuesString[i]);
        }
        return meanValues;
    }

    public String getSvmTrainOptions(){
        return sharedPreferences.getString("key_svmTrainOptions", "-t 0 ");
    }

    public int getK(){
        return Integer.valueOf(sharedPreferences.getString("key_K", "20"));
    }

    public int getN(){
        return Integer.valueOf(sharedPreferences.getString("key_N", "25"));
    }

    public int getFaceSize(){
        return Integer.valueOf(sharedPreferences.getString("key_faceSize", "160"));
    }

    public int getTensorFlowInputSize(){
        return Integer.valueOf(sharedPreferences.getString("key_inputSize", "160"));
    }

    public int getTensorFlowInputChannels(){
        return Integer.valueOf(sharedPreferences.getString("key_inputChannels", "3"));
    }

    public int getTensorFlowImageMean(){
        return Integer.valueOf(sharedPreferences.getString("key_imageMean", "128"));
    }

    public int getTensorFlowImageStd(){
        return Integer.valueOf(sharedPreferences.getString("key_imageStd", "128"));
    }

    public int getTensorFlowOutputSize(){
        return Integer.valueOf(sharedPreferences.getString("key_outputSize", "128"));
    }

    public String getTensorFlowInputLayer(){
        return sharedPreferences.getString("key_inputLayer", "input");
    }

    public String getTensorFlowOutputLayer(){
        return sharedPreferences.getString("key_outputLayer", "embeddings");
    }

    public String getTensorFlowModelFile(){
        return sharedPreferences.getString("key_modelFileTensorFlow", "facenet.pb");
    }

    public float getPCAThreshold(){
        return Float.valueOf(sharedPreferences.getString("key_pca_threshold", "0.98f"));
    }

    public String getFaceCascadeFile(){
        return sharedPreferences.getString("key_face_cascade_file", resources.getString(R.string.haarcascade_alt2));
    }

    public String getLefteyeCascadeFile(){
        return sharedPreferences.getString("key_lefteye_cascade_file", resources.getString(R.string.haarcascade_lefteye));
    }

    public String getRighteyeCascadeFile(){
        return sharedPreferences.getString("key_righteye_cascade_file", resources.getString(R.string.haarcascade_righteye));
    }

    public double getDetectionScaleFactor(){
        return Double.parseDouble(sharedPreferences.getString("key_scaleFactor", "1.1"));
    }

    public int getDetectionMinNeighbors(){
        return Integer.parseInt(sharedPreferences.getString("key_minNeighbors", "3"));
    }

    public int getDetectionFlags(){
        return Integer.parseInt(sharedPreferences.getString("key_flags", "2"));
    }

    public boolean getDetectionMethod(){
        return sharedPreferences.getBoolean("key_detection_method", true);
    }
}
