import os

counts={"0":0,"1":0}
for file in os.listdir("./Dataset"):
  if file.endswith(".txt") and file!='classes.txt':
    f = open("./Dataset/"+file)
    lines=f.readlines()
    
    for line in lines:
      counts[''+line[0]]+=1
      
    
print(counts)