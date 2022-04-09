from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

def recognize(embeddings,names):
	
	le = LabelEncoder()
	labels = le.fit_transform(names)
	recognizer = SVC(C=1.0, kernel="linear", probability=True)
	recognizer.fit(embeddings, names)

	return le,recognizer