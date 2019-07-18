#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from core.model import ModelWrapper
from maxfw.core import MAX_API, PredictAPI

import os
from flask_restplus import fields
from werkzeug.datastructures import FileStorage


UPLOAD_FOLDER = './assets'


label_prediction = MAX_API.model('LabelPrediction', {
    'label_id': fields.String(required=False, description='Class label identifier'),
    'label': fields.String(required=True, description='Class label'),
    'probability': fields.Float(required=True)
})

predict_response = MAX_API.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'predictions': fields.List(fields.Nested(label_prediction), description='Predicted labels and probabilities')
})

# set up parser for image input data
video_parser = MAX_API.parser()
video_parser.add_argument('video', type=FileStorage, location='files', required=True,
                          help="MPEG-4 video file to run predictions on")


@MAX_API.route('/predict')
class ModelPredictAPI(PredictAPI):

    model_wrapper = ModelWrapper()

    @MAX_API.doc('predict')
    @MAX_API.expect(video_parser)
    @MAX_API.marshal_with(predict_response)
    def post(self):
        """Make a prediction given input data"""
        result = {'status': 'error'}

        #  Take video save it into directory
        args = video_parser.parse_args()
        video_data = args['video']
        filepath = os.path.join(UPLOAD_FOLDER, video_data.filename)
        video_data.save(filepath)

        #  Run predict function on file
        preds = self.model_wrapper.predict(filepath)
        label_preds = [{'label_id': p[0], 'label': p[1], 'probability': p[2]} for p in [x for x in preds]]
        result['predictions'] = label_preds
        result['status'] = 'ok'

        return result
