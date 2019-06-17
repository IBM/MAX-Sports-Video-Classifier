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

# Application settings

# Flask settings 
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False
SWAGGER_UI_DOC_EXPANSION = 'none'

# API metadata
API_TITLE = 'MAX Sports Video Classifier'
API_DESC = 'Classify sporting activities in videos.'
API_VERSION = '1.1.0'

# default model
MODEL_NAME = 'C3D'
DEFAULT_MODEL_PATH = 'assets'
DEFAULT_MODEL_DIR = MODEL_NAME.lower()

# model batch size (# video frames per batch)
BATCH_SIZE = 10
# Images are cropped to (CROP_SIZE, CROP_SIZE)
CROP_SIZE = 112
CHANNELS = 3
# Number of frames to extract for a video clip
NUM_FRAMES_PER_CLIP = 16

MODEL_LICENSE = 'MIT'

MODEL_META_DATA = {
    'id': '{}-tf'.format(MODEL_NAME.lower()),
    'name': '{} TensorFlow Model'.format(MODEL_NAME),
    'description': '{} TensorFlow video classification model trained on the Sports1m dataset'.format(MODEL_NAME),
    'type': 'video_classification',
    'license': '{}'.format(MODEL_LICENSE)
}
