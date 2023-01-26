from flask import Blueprint, make_response, jsonify, request
from PIL import Image
from api.model import make_prediction, model, prepare_image
import io
from api.constants import *
from api.types import *

blueprint = Blueprint("blueprint",__name__)

@blueprint.route('/gender/classify', methods=["POST"])
def classify_gender():
    if request.method == "POST":
        if request.files.get("image"):
            try:
                image = request.files.get("image")
                ext = "."+str(image.filename).split('.')[-1].lower()
                if not ext in allowed_extensions:
                    return make_response(jsonify({
                        "ok" : False,
                        "error": {
                            "field" : 'image',
                            "message" : f'Only images with extensions ({", ".join(allowed_extensions)}) are allowed.'
                        }
                    })), 200
                image = Image.open(io.BytesIO(image.read()))
                image = prepare_image(image, target=(96, 96))
                res = make_prediction(model, image)
                return make_response(jsonify({
                    'ok': True,
                    'prediction': res.to_json() 
                })), 200
            except Exception as e:
                print(e)
                return make_response(jsonify({
                    "ok": False,
                    "error":{
                        "field": 'server',
                        'message':  "Something went wrong on the server."
                    }
                })), 200
        else:
            return make_response(jsonify({
                "ok" : False,
                "error": {
                    "field" : 'image',
                    "message" : f'There was no image in your request.'
                }
            })), 200
    else:
        return make_response(jsonify({
                "ok" : False,
                "error": {
                    "field" : 'method',
                    "message" : f'The request method that is allowed is "POST".'
                }
            })), 200
        
