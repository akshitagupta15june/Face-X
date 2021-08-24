import bpy
import cv2
import time
import numpy

class OpenCVAnimOperator(bpy.types.Operator):
    bl_idname = "wm.opencv_operator"
    bl_label = "OpenCV Animation Operator"
    
    face_detect_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    landmark_model_path = "./data/lbfmodel.yaml"

    fm = cv2.face.createFacemarkLBF()
    fm.loadModel(landmark_model_path)
    cas = cv2.CascadeClassifier(face_detect_path)
    
    _timer = None
    _cap  = None
    stop = False
  
    width = 640
    height = 480
    
    model_points = numpy.array([
                                (0.0, 0.0, 0.0),            
                                (0.0, -330.0, -65.0),        
                                (-225.0, 170.0, -135.0),     
                                (225.0, 170.0, -135.0),     
                                (-150.0, -150.0, -125.0),   
                                (150.0, -150.0, -125.0)     
                            ], dtype = numpy.float32)
    camera_matrix = numpy.array(
                            [[height, 0.0, width/2],
                            [0.0, height, height/2],
                            [0.0, 0.0, 1.0]], dtype = numpy.float32
                            )
                            
    def smooth_value(self, name, length, value):
        if not hasattr(self, 'smooth'):
            self.smooth = {}
        if not name in self.smooth:
            self.smooth[name] = numpy.array([value])
        else:
            self.smooth[name] = numpy.insert(arr=self.smooth[name], obj=0, values=value)
            if self.smooth[name].size > length:
                self.smooth[name] = numpy.delete(self.smooth[name], self.smooth[name].size-1, 0)
        sum = 0
        for val in self.smooth[name]:
            sum += val
        return sum / self.smooth[name].size

    def get_range(self, name, value):
        if not hasattr(self, 'range'):
            self.range = {}
        if not name in self.range:
            self.range[name] = numpy.array([value, value])
        else:
            self.range[name] = numpy.array([min(value, self.range[name][0]), max(value, self.range[name][1])] )
        val_range = self.range[name][1] - self.range[name][0]
        if val_range != 0:
            return (value - self.range[name][0]) / val_range
        else:
            return 0.0
        
    def modal(self, context, event):

        if (event.type in {'RIGHTMOUSE', 'ESC'}) or self.stop == True:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            self.init_camera()
            _, image = self._cap.read()
            
            faces = self.cas.detectMultiScale(image, 
                scaleFactor=1.05,  
                minNeighbors=3, 
                flags=cv2.CASCADE_SCALE_IMAGE, 
                minSize=(int(self.width/5), int(self.width/5)))
            
            if type(faces) is numpy.ndarray and faces.size > 0: 
                biggestFace = numpy.zeros(shape=(1,4))
                for face in faces:
                    if face[2] > biggestFace[0][2]:
                        print(face)
                        biggestFace[0] = face
         
                _, landmarks = self.fm.fit(image, faces=biggestFace)
                for mark in landmarks:
                    shape = mark[0]
                    
                    image_points = numpy.array([shape[30],     
                                                shape[8],     
                                                shape[36],     
                                                shape[45],   
                                                shape[48],    
                                                shape[54]
                                            ], dtype = numpy.float32)
                 
                    dist_coeffs = numpy.zeros((4,1))
                 
                    if hasattr(self, 'rotation_vector'):
                        (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points, 
                            image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, 
                            rvec=self.rotation_vector, tvec=self.translation_vector, 
                            useExtrinsicGuess=True)
                    else:
                        (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points, 
                            image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, 
                            useExtrinsicGuess=False)
                 
                    if not hasattr(self, 'first_angle'):
                        self.first_angle = numpy.copy(self.rotation_vector)
                    
                    bones = bpy.data.objects["RIG-Vincent"].pose.bones
                    
                    bones["head_fk"].rotation_euler[0] = self.smooth_value("h_x", 5, (self.rotation_vector[0] - self.first_angle[0])) / 1   
                    bones["head_fk"].rotation_euler[2] = self.smooth_value("h_y", 5, -(self.rotation_vector[1] - self.first_angle[1])) / 1.5  
                    bones["head_fk"].rotation_euler[1] = self.smooth_value("h_z", 5, (self.rotation_vector[2] - self.first_angle[2])) / 1.3  
                    
                    bones["head_fk"].keyframe_insert(data_path="rotation_euler", index=-1)
                    
                    bones["mouth_ctrl"].location[2] = self.smooth_value("m_h", 2, -self.get_range("mouth_height", numpy.linalg.norm(shape[62] - shape[66])) * 0.06 )
                    bones["mouth_ctrl"].location[0] = self.smooth_value("m_w", 2, (self.get_range("mouth_width", numpy.linalg.norm(shape[54] - shape[48])) - 0.5) * -0.04)
                    
                    bones["mouth_ctrl"].keyframe_insert(data_path="location", index=-1)
                    
                    bones["brow_ctrl_L"].location[2] = self.smooth_value("b_l", 3, (self.get_range("brow_left", numpy.linalg.norm(shape[19] - shape[27])) -0.5) * 0.04)
                    bones["brow_ctrl_R"].location[2] = self.smooth_value("b_r", 3, (self.get_range("brow_right", numpy.linalg.norm(shape[24] - shape[27])) -0.5) * 0.04)
                    
                    bones["brow_ctrl_L"].keyframe_insert(data_path="location", index=2)
                    bones["brow_ctrl_R"].keyframe_insert(data_path="location", index=2)
                    
                    l_open = self.smooth_value("e_l", 2, self.get_range("l_open", -numpy.linalg.norm(shape[48] - shape[44]))  )
                    r_open = self.smooth_value("e_r", 2, self.get_range("r_open", -numpy.linalg.norm(shape[41] - shape[39]))  )
                    eyes_open = (l_open + r_open) / 2.0 
                    bones["eyelid_up_ctrl_R"].location[2] =   -eyes_open * 0.025 + 0.005
                    bones["eyelid_low_ctrl_R"].location[2] =  eyes_open * 0.025 - 0.005
                    bones["eyelid_up_ctrl_L"].location[2] =   -eyes_open * 0.025 + 0.005
                    bones["eyelid_low_ctrl_L"].location[2] =  eyes_open * 0.025 - 0.005
                    
                    bones["eyelid_up_ctrl_R"].keyframe_insert(data_path="location", index=2)
                    bones["eyelid_low_ctrl_R"].keyframe_insert(data_path="location", index=2)
                    bones["eyelid_up_ctrl_L"].keyframe_insert(data_path="location", index=2)
                    bones["eyelid_low_ctrl_L"].keyframe_insert(data_path="location", index=2)
                    
                    for (x, y) in shape:
                        cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
            
            for (x,y,w,h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),1)
            
            cv2.imshow("Output", image)
            cv2.waitKey(1)

        return {'PASS_THROUGH'}
    
    def init_camera(self):
        if self._cap == None:
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(1.0)
    
    def stop_playback(self, scene):
        print(format(scene.frame_current) + " / " + format(scene.frame_end))
        if scene.frame_current == scene.frame_end:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        
    def execute(self, context):
        bpy.app.handlers.frame_change_pre.append(self.stop_playback)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        cv2.destroyAllWindows()
        self._cap.release()
        self._cap = None

def register():
    bpy.utils.register_class(OpenCVAnimOperator)

def unregister():
    bpy.utils.unregister_class(OpenCVAnimOperator)

if __name__ == "__main__":
    register()

