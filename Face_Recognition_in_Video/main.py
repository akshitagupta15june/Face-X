from tkinter import *
from user_interface import *
import pathlib
import os
import glob
from tkinter import messagebox
# from PIL import Image

try:
    directory="Videos"
    parent_dir=pathlib.Path().resolve()
    path=os.path.join(parent_dir,directory)
    os.mkdir(path)
except:
    pass

try:
    directory="Images"
    parent_dir=pathlib.Path().resolve()
    path=os.path.join(parent_dir,directory)
    os.mkdir(path)
except:
    pass


windows=Tk()
windows.title("Face Recognition in Video")
windows.config(padx=50,pady=50)

images, class_name=image_list()
encode_list_known=findEncodings(images)
#canvas
canvas=Canvas(width=200,height=224)
img=PhotoImage(file="face_image.png")
canvas.create_image(100,112,image=img)
canvas.grid(row=0,column=0,columnspan=2)

#label
label1=Label(text="Known Images:")
label1.grid(row=1,column=0)

#List box
def list_Box(evt):
    # print(str(list_box.get(list_box.curselection())))
    pass
list_box=Listbox(height=5,width=30)
list_box.bind("<<ListboxSelect>>",list_Box)
[list_box.insert(END,name) for name in class_name]
list_box.grid(row=1,column=1)

label2=Label(text="Add image to known images list:")
label2.grid(row=2,column=0)

#Entry
entry1=Entry(width=30)
entry1.insert(END,"Enter path to image file")
entry1.grid(row=2,column=1)

def button1_click():
    path=pathlib.Path(entry1.get())
    file_name=os.path.basename(entry1.get())
    target = f"{pathlib.Path().resolve()}/Images/{file_name}"
    path.rename(target)
    new_file_name=os.path.splitext(file_name)[0]
    class_name.append(new_file_name)
    # list_box.insert(new_file_name)


button1=Button(text="Add",width=30,command=button1_click)
button1.grid(row=3,column=1)

label3=Label(text="Video to be checked")
label3.grid(row=4,column=0)

#Entry
entry2=Entry(width=30)
entry2.insert(END,"Enter path to video file")
entry2.grid(row=4,column=1)


def button2_click():
    #add file to Videos folder
    path = pathlib.Path(entry2.get())
    file_name = os.path.basename(entry2.get())
    target = f"{pathlib.Path().resolve()}/Videos/{file_name}"
    path=path.rename(target)

    video_capture = cv2.VideoCapture(target)
    found=face_recog(video_capture, encode_list_known, class_name)
    if found:
        messagebox.showinfo(title="Face recognition in Video Result",message="The image added is found in video")
    else:
        messagebox.showinfo(title="Face recognition in Video Result",message="The image added is not found in video")


button2=Button(text="Add",width=30,command=button2_click)
button2.grid(row=5,column=1)

windows.mainloop()
