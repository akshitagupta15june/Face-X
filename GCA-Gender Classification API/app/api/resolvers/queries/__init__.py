from ariadne import QueryType
from api.types import * 


query = QueryType()
@query.field("meta")
def meta_resolver(obj, info):
   return  Meta(
        description = "classifying gender based on the face of a human being, (vgg16).",
        programmer = "@crispengari",
        language = "python",
        main = "computer vision (cv)",
        libraries = ["tensorflow"]
    ).to_json()