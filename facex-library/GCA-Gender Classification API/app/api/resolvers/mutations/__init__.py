from ariadne import MutationType
from api.types import * 
from api.model import *
from PIL import Image
from api.constants import *
import io


mutation = MutationType()
@mutation.field("classifyGender")
def classify_gender_resolver(obj, info, input):
   try:
        ext = "."+str(input['image'].filename).split('.')[-1].lower()
        if not ext in allowed_extensions:
            return {
                "ok" : False,
                "error": {
                    "field" : 'image',
                    "message" : f'Only images with extensions ({", ".join(allowed_extensions)}) are allowed.'
                }
            }
        image = input['image'].read()
        image = Image.open(io.BytesIO(image))
        image = prepare_image(image, target=(96, 96))
        res = make_prediction(model, image)
        return {
           'ok': True,
           'prediction': res.to_json() 
        }
   except Exception as e:
       print(e)
       return {
           "ok": False,
           "error":{
               "field": 'server',
               'message':  "Something went wrong on the server."
           }
       }