import pytest
import requests


def test_response():

    model_endpoint = 'http://localhost:5000/model/predict'
    file_name = 'basketball.mp4'  # due to a weird issue with the way model.py saves the uploaded video `file_form` can
                                  # only contain the file name, not the path. This issue does not exist when using CURL
    file_path = 'assets/' + file_name

    with open(file_path, 'rb') as file:
        file_form = {'video': (file_name, file, 'video/mp4')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200

    response = r.json()

    assert response['status'] == 'ok'

    # Check the label id for each prediction
    assert response['predictions'][0]['label_id'] == '370'
    assert response['predictions'][1]['label_id'] == '367'
    assert response['predictions'][2]['label_id'] == '369'

    # Check the label names for each prediction
    assert response['predictions'][0]['label'] == 'streetball'
    assert response['predictions'][1]['label'] == 'basketball'
    assert response['predictions'][2]['label'] == '3x3 (basketball)'

    # Check that the probability has not fallen
    assert response['predictions'][0]['probability'] > 0.25
    assert response['predictions'][0]['probability'] > 0.2
    assert response['predictions'][0]['probability'] > 0.15

    # Make sure that the predictions are sorted by probability
    assert response['predictions'][0]['probability'] > response['predictions'][1]['probability']
    assert response['predictions'][1]['probability'] > response['predictions'][2]['probability']


if __name__ == '__main__':
    pytest.main([__file__])
