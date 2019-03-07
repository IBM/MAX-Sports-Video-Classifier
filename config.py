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
