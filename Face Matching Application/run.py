import os 

def execute_system():
    bash1 = 'python src/01_generate_img_pkl.py'
    bash2 = 'python src/02_feature_extractor.py'

    os.system(bash1)
    os.system(bash2)
    print('Executed successfully!! Now run app.py')

if __name__ == '__main__':
    execute_system()