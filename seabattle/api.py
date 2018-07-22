# coding: utf-8

from __future__ import unicode_literals

import json
import logging
import sys

from flask import Flask, request

from seabattle import dialog_manager as dm
from seabattle import session


logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
log = logging.getLogger(__name__)


@app.route('/', methods=['POST'])
def main():
    log.info('Request: %r', request.json)

    response = {
        'version': request.json['version'],
        'session': request.json['session'],
    }
    json_body = request.json

    user_id = json_body['session']['user_id']
    session_obj = session.get(user_id)
    dm_obj = dm.DialogManager(session_obj)

    message = json_body['request']['command'].strip()
    if not message:
        message = json_body['request']['original_utterance']

    dmresponse = dm_obj.handle_message(message)
    response['response'] = {
        'text': dmresponse.text,
        'end_session': dmresponse.end_session,
    }
    if dmresponse.tts is not None:
        response['response']['tts'] = dmresponse.tts

    log.info('Response: %r', response)
    return json.dumps(response)
