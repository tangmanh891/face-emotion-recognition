import base64
import io
import unittest

import cv2
import numpy as np

import app as app_module


class FakeDetector:
    def analyze_faces(self, _image):
        return [
            {
                'x': 1,
                'y': 2,
                'w': 10,
                'h': 12,
                'emotion': 'Vui ve',
                'confidence': 98.765,
                'probabilities': {
                    'Vui ve': 98.765,
                    'Buon ba': 1.235,
                },
            }
        ]

    def draw_results(self, image, _results):
        return image


class EmptyDetector:
    """Detector that finds no faces — exercises the empty-result branch."""

    def analyze_faces(self, _image):
        return []


class CrashingDetector:
    """Detector that raises — exercises the generic 500 path."""

    def analyze_faces(self, _image):
        raise RuntimeError('boom')

    def draw_results(self, *_):  # pragma: no cover
        raise RuntimeError('boom')


def _make_jpeg_bytes(size=(32, 32)):
    image = np.zeros((*size, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', image)
    assert ok
    return encoded.tobytes()


class AppApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()
        self.original_detector = app_module.detector
        app_module.app.config['TESTING'] = True

    def tearDown(self):
        app_module.detector = self.original_detector

    def test_health_endpoint(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['status'], 'ok')
        self.assertIn('model_loaded', payload)
        self.assertIn('model_path', payload)
        self.assertIn('model_found', payload)

    def test_index_renders(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_detect_requires_file(self):
        app_module.detector = FakeDetector()
        response = self.client.post('/detect', data={}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())

    def test_detect_rejects_unsupported_mimetype(self):
        app_module.detector = FakeDetector()
        data = {'image': (io.BytesIO(b'fakegif'), 'pic.gif', 'image/gif')}
        response = self.client.post('/detect', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)

    def test_detect_rejects_unreadable_image(self):
        app_module.detector = FakeDetector()
        data = {'image': (io.BytesIO(b'not actually an image'), 'pic.jpg', 'image/jpeg')}
        response = self.client.post('/detect', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)

    def test_detect_rejects_oversized_image(self):
        app_module.detector = FakeDetector()
        # Just above the 5MB limit; Werkzeug aborts before reading the body.
        big = b'\x00' * (app_module.MAX_IMAGE_SIZE_BYTES + 1024)
        data = {'image': (io.BytesIO(big), 'big.jpg', 'image/jpeg')}
        response = self.client.post('/detect', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 413)

    def test_detect_no_faces_response(self):
        app_module.detector = EmptyDetector()
        data = {'image': (io.BytesIO(_make_jpeg_bytes()), 'face.jpg')}
        response = self.client.post('/detect', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual(payload['faces_count'], 0)
        self.assertEqual(payload['faces'], [])
        self.assertIn('message', payload)

    def test_detect_success_includes_probabilities(self):
        app_module.detector = FakeDetector()
        data = {'image': (io.BytesIO(_make_jpeg_bytes()), 'face.jpg')}
        response = self.client.post('/detect', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual(payload['faces_count'], 1)
        self.assertIn('probabilities', payload['faces'][0])
        self.assertTrue(payload['image'].startswith('data:image/jpeg;base64,'))

    def test_detect_returns_500_when_no_detector(self):
        app_module.detector = None
        # Patch out lazy init so the global stays None.
        original_init = app_module.init_detector
        app_module.init_detector = lambda: None
        try:
            data = {'image': (io.BytesIO(_make_jpeg_bytes()), 'face.jpg')}
            response = self.client.post('/detect', data=data, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 500)
        finally:
            app_module.init_detector = original_init

    def test_detect_returns_500_on_unexpected_error(self):
        app_module.detector = CrashingDetector()
        data = {'image': (io.BytesIO(_make_jpeg_bytes()), 'face.jpg')}
        response = self.client.post('/detect', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 500)
        # Generic message — internal error text must not leak.
        self.assertNotIn('boom', response.get_data(as_text=True))


class AppBase64Tests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()
        self.original_detector = app_module.detector

    def tearDown(self):
        app_module.detector = self.original_detector

    def test_detect_base64_rejects_invalid_payload(self):
        app_module.detector = FakeDetector()
        response = self.client.post('/detect_base64', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())

    def test_detect_base64_rejects_non_dict_payload(self):
        app_module.detector = FakeDetector()
        response = self.client.post('/detect_base64', json='not-an-object')
        self.assertEqual(response.status_code, 400)

    def test_detect_base64_rejects_bad_base64(self):
        app_module.detector = FakeDetector()
        response = self.client.post('/detect_base64', json={'image': 'data:image/jpeg;base64,@@@'})
        self.assertEqual(response.status_code, 400)

    def test_detect_base64_success(self):
        app_module.detector = FakeDetector()
        image_b64 = base64.b64encode(_make_jpeg_bytes()).decode('utf-8')
        response = self.client.post(
            '/detect_base64',
            json={'image': f'data:image/jpeg;base64,{image_b64}'},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual(payload['faces_count'], 1)
        # Webcam endpoint does NOT echo the annotated image.
        self.assertNotIn('image', payload)

    def test_detect_base64_accepts_raw_string_without_prefix(self):
        app_module.detector = FakeDetector()
        image_b64 = base64.b64encode(_make_jpeg_bytes()).decode('utf-8')
        response = self.client.post('/detect_base64', json={'image': image_b64})
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
