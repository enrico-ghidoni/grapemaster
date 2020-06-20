import flask
import googleapiclient
import json
import numpy as np

from googleapiclient import discovery


def match_variety(request):
    # init google api client
    service = googleapiclient.discovery.build('ml', 'v1')

    parameters = request.get_json()['queryResult']['parameters']
    if 'description' in parameters:
        # user has opted for classification based on description
        user_description = parameters['description']        
        request_body = {'instances': [user_description]}

        service_name = 'projects/sac-wine-analysis/models/grapemaster_description_clf'
    else:
        # user has opted for classification based on adjectives
        # flatten the parameters arrays into one and ensure no duplicate entries
        flattened = [adj for key in parameters.keys() for adj in parameters[key]]
        user_adjectives = list(set(flattened))

        request_body = {'instances': user_adjectives}

        service_name = 'projects/sac-wine-analysis/models/grapemaster_adjective_clf'

    model_prediction = service.projects().predict(
            name=service_name,
            body=request_body
        ).execute()

    predicted_variety = model_prediction['predictions'][0]

    response = {'fulfillmentMessages': []}
    response['fulfillmentMessages'].append({'text': {'text': [f'Based on your description, you may like {predicted_variety}']}})

    return json.dumps(response), 200
