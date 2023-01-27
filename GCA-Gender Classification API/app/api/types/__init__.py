class Meta:
    def __init__(self,
                 programmer:str,
                 main: str,
                 description: str,
                 language: str,
                 libraries: list
                 ):
        self.programmer = programmer
        self.main = main
        self.description = description
        self.language = language
        self.libraries = libraries
        
    def to_json(self):
        return{
            "programmer": self.programmer,
            "main": self.main,
            "description": self.description,
            "language": self.language,
            "libraries": self.libraries,
        }
        

class Prediction:
    def __init__(self, label:int, probability:float, class_name: str) -> None:
        self.label = label
        self.probability = probability
        self.class_name = class_name
        
    def __repr__(self) -> str:
        return f"<{self.class_name}>"
    
    def __str__(self) -> str:
        return f"<{self.class_name}>"
    
    def to_json(self):
        return {
            "label": int(self.label),
            "probability": float(self.probability),
            "className": self.class_name,
        }
    
class Response:
    def __init__(self, top_prediction: Prediction, predictions:list) -> None:
        self.predictions = predictions
        self.top_prediction = top_prediction
        
    def __repr__(self) -> str:
        return f""
    
    def __str__(self) -> str:
        return f""
    
    def to_json(self):
        return{
           "predictions": [item.to_json() for item in self.predictions],
            "topPrediction": self.top_prediction.to_json()
        }