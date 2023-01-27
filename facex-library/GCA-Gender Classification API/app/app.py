from api import app
from flask import make_response, jsonify, request, json
from api.blueprints import blueprint
from ariadne import  load_schema_from_path, make_executable_schema, graphql_sync, upload_scalar, combine_multipart_data
from ariadne.constants import PLAYGROUND_HTML
from api.resolvers.queries import query
from api.resolvers.mutations import mutation
from api.types import Meta


type_defs = load_schema_from_path("schema/schema.graphql")
schema = make_executable_schema(
    type_defs, [upload_scalar, query, mutation, ]
)

app.register_blueprint(blueprint, url_prefix="/api")

class AppConfig:
    PORT = 3001
    DEBUG = False
    
    
@app.route('/', methods=["GET"])
def meta():
    _meta = Meta(
        description = "classifying gender based on the face of a human being, (vgg16).",
        programmer = "@crispengari",
        language = "python",
        main = "computer vision (cv)",
        libraries = ["tensorflow"]
    )
    return make_response(jsonify(_meta.to_json())), 200


@app.route("/graphql", methods=["GET"], )
def graphql_playground():
    return PLAYGROUND_HTML, 200

@app.route("/graphql", methods=["POST"])
def graphql_server():
    if  request.content_type.startswith("multipart/form-data" ):
         data = combine_multipart_data(
            json.loads(request.form.get("operations")),
            json.loads(request.form.get("map")),
            dict(request.files)
        )
    else:
        data =  request.get_json()
    success, result = graphql_sync(
        schema,
        data,
        context_value=request,
        debug= AppConfig.DEBUG
    )
    return jsonify(result), 200 if success else 400

if __name__ == "__main__":
    app.run(debug=AppConfig().DEBUG, port=AppConfig().PORT, )