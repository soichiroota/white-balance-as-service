import os
import json
import io

import responder
import numpy as np

from white_balance import (
    load_img, save_img, stretch_to_8bit,
    WhiteBalancer
)


env = os.environ
DEBUG = env['DEBUG'] in ['1', 'True', 'true']
ALGO = env.get('ALGO')
FORMAT = env.get('FORMAT')

api = responder.API(debug=DEBUG)
cur_dir = os.path.dirname(__file__)
white_balancer = WhiteBalancer(
    ALGO,
    model_folder=os.path.join(
        cur_dir, 'model'
    )
)


def balance(bytes_, format_='PNG'):
    img_array = load_img(bytes_)
    balanced_img_array = white_balancer.balance(img_array)
    return save_img(balanced_img_array, format_=format_)


@api.route("/")
async def encode(req, resp):
    body = await req.content 
    resp.content = balance(body, format_=FORMAT)


if __name__ == "__main__":
    api.run()