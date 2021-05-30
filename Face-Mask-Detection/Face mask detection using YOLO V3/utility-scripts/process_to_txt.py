import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
# Create and/or truncate train.txt 
file_train = open(current_dir+'/train.txt', 'w')  
current_dir += '/train'


for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write(current_dir + "/" + title + '.jpg' + "\n")
	
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpeg")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write(current_dir + "/" + title + '.jpeg' + "\n")
	
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.png")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write(current_dir + "/" + title + '.png' + "\n")

for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.gif")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write(current_dir + "/" + title + '.gif' + "\n")

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)	

# Create and/or truncate valid.txt 
file_train = open(current_dir+'/valid.txt', 'w')  
current_dir += '/valid'




for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write(current_dir + "/" + title + '.jpg' + "\n")
        
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpeg")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write(current_dir + "/" + title + '.jpeg' + "\n")
        
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.png")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write(current_dir + "/" + title + '.png' + "\n")

for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.gif")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write(current_dir + "/" + title + '.gif' + "\n")

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)	
# Create and/or truncate test.txt 
file_train = open(current_dir+'/test.txt', 'w') 
current_dir += '/test'

 



for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write(current_dir + "/" + title + '.jpg' + "\n")
        
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpeg")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write(current_dir + "/" + title + '.jpeg' + "\n")
        
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.png")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write(current_dir + "/" + title + '.png' + "\n")

for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.gif")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write(current_dir + "/" + title + '.gif' + "\n")