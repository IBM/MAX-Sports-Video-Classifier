import pytest
import pycurl
import io
import json


def test_response():
    c = pycurl.Curl()
    b = io.BytesIO()
    c.setopt(pycurl.URL, 'http://localhost:5000/model/predict')
    c.setopt(pycurl.HTTPHEADER, ['Accept:application/json', 'Content-Type: multipart/form-data'])
    c.setopt(pycurl.HTTPPOST, [('video', (pycurl.FORM_FILE, "assets/basketball.mp4"))])
    c.setopt(pycurl.WRITEFUNCTION, b.write)
    c.perform()
    assert c.getinfo(pycurl.RESPONSE_CODE) == 200
    c.close()

    response = b.getvalue()
    response = json.loads(response)

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
